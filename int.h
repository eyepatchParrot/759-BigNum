#ifndef INT_H
#define INT_H

#include <cstdint>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <assert.h>
#include <iomanip>

#include <gmp.h>
#include <gmpxx.h>

#include "cksum.h"

/*
 * It doesn't make any sense to talk about a fixed block size because different
 * ops should be tiled differently.
 *
 * Need to separate capacity from size to keep integers small after many adds, etc.
 */
namespace GPU {
  // TODO make sure everything is OK. I think this doesn't match the instrinsic I'm using.
  using Limb = uint64_t;

static __global__ void d_cksum(uint16_t* sum, const uint8_t* data, int n_bytes) {
  *sum = cksum(data, n_bytes);
}

static __device__ int tid() { return blockIdx.x * blockDim.x + threadIdx.x; }


template <typename T>
static T d_read(T* d) {
  T t;
  cudaMemcpy(&t, d, sizeof(T), cudaMemcpyDeviceToHost);
  return t;
}

__global__ void g_carry_reduce_limb_block(Limb* p, Limb* g, const Limb* a, const int a_sz, const Limb* b, const int b_sz) ;
__global__ void g_reduce(Limb* p, Limb* g, const int n, const int block_sz) ;
__global__ void g_sweep(Limb* p, Limb* g, const int n, const int block_sz) ;
__global__ void g_add(Limb* c,  const Limb* a, const int a_sz, const Limb* b, const int b_sz, const Limb* p, const Limb *g) ;
__global__ void g_times(Limb* c, const Limb* a, const int a_n, const Limb* b, int b_n, Limb* carry) ;
__global__ void g_add_serial(Limb* c,  const Limb* a, const int a_sz, const Limb* b, const int b_sz, Limb* carry) ;
__global__ void g_carry_serial(Limb* c, int n_c, Limb* carry, int block_sz);




class Int {
  public:
  static constexpr int add_block_sz = 256;
  static constexpr int carry_block_sz = 256;
  static constexpr int times_block_sz = 256;

  private:
  // TODO consider including some extra space to avoid re-allocation
  // probably have a minimum size.

  template <typename T>
  class d_Val {
    T v;
    T* d_v;
  public:
    d_Val() {
      cudaMalloc(&d_v, sizeof(T));
    }
    d_Val(T _v) : v(_v) {
      cudaMalloc(&d_v, sizeof(T));
      cudaMemcpy(d_v, &v, sizeof(T), cudaMemcpyHostToDevice);
    }
    d_Val(T* d_v) : d_v(d_v) {}

    ~d_Val() { cudaFree(d_v); }
    operator T*() {
      return d_v;
    }
    operator T() {
      cudaMemcpy(&v, d_v, sizeof(T), cudaMemcpyDeviceToHost);
      return v;
    }
  };

  Int add_serial(const Int& other) {
    // I need some way to not allocate and free the carry every time.
    const Int& a = this->n_limbs >= other.n_limbs ? *this : other;
    const Int& b = this->n_limbs >= other.n_limbs ? other : *this;

    int n_blocks = (a.n_limbs - 1 + add_block_sz) / add_block_sz;

    Int c, carry;
    c.resize(a.n_limbs + 1);
    carry.resize(n_blocks);
    //Int c = std::move(Int().reserve(a.n_limbs));

    // assumes fewer than 1024 * add_block_sz limbs
    g_add_serial<<<n_blocks, add_block_sz>>>(c.d_limbs, a.d_limbs, a.n_limbs, b.d_limbs, b.n_limbs, carry.d_limbs);
    g_carry_serial<<<1, 1>>>(c.d_limbs, a.n_limbs, carry.d_limbs, add_block_sz);
    if (d_read(c.d_limbs + a.n_limbs) == 0) c.resize(a.n_limbs);

    return c;
  }

  Int add(const Int& other) {
    // I need some way to not allocate and free the carry every time.
    const Int& a = this->n_limbs >= other.n_limbs ? *this : other;
    const Int& b = this->n_limbs >= other.n_limbs ? other : *this;

    int n_blocks = (a.n_limbs - 1 + add_block_sz) / add_block_sz;

    Int c, p, g;
    c.resize(a.n_limbs + 1);
    p.resize(n_blocks);
    g.resize(n_blocks);

    g_carry_reduce_limb_block<<<n_blocks, add_block_sz>>>(p.d_limbs, g.d_limbs, a.d_limbs, a.n_limbs, b.d_limbs, b.n_limbs);
    // TODO make sure we're not off by one on n_limbs
    for (int k = 1; add_block_sz * k - 1 < n_blocks; k *= add_block_sz)
      g_reduce<<<
        (n_blocks - 1 + add_block_sz * k) / k / add_block_sz,
        add_block_sz>>>
          (p.d_limbs, g.d_limbs, c.n_limbs, k); 
    for (int k = (n_blocks - 1 + add_block_sz) / add_block_sz; k >= 1; k /= add_block_sz)
      g_sweep<<<
        (n_blocks - 1 + add_block_sz * k) / k / add_block_sz,
        add_block_sz>>>(p.d_limbs, g.d_limbs, n_blocks, k);

    g_add<<<n_blocks, add_block_sz>>>(c.d_limbs, a.d_limbs, a.n_limbs, b.d_limbs, b.n_limbs, p.d_limbs, g.d_limbs);
    if (d_read(c.d_limbs + a.n_limbs) == 0) c.resize(a.n_limbs);

    return c;
  }

  Int times(const Int& other) {
    const Int& a = this->n_limbs >= other.n_limbs ? *this : other;
    const Int& b = this->n_limbs >= other.n_limbs ? other : *this;

    // a_i <= a_n - 1, b_i <= b_n - 1 ==> a_i + b_i < a_n + b_n - 1
    int c_n = a.n_limbs + b.n_limbs - 1;

    int n_blocks = (c_n - 1 + times_block_sz) / times_block_sz;
    Int c, carry;
    c.resize(c_n);
    carry.resize(c_n + 2);
    g_times<<<n_blocks, times_block_sz>>>(c.d_limbs, a.d_limbs, a.n_limbs, b.d_limbs, b.n_limbs, carry.d_limbs);
    if (d_read(carry.d_limbs + c_n + 1) == 0) {
      if (d_read(carry.d_limbs + c_n) == 0) carry.resize(c_n);
      else carry.resize(c_n+1);
    } else carry.resize(c_n+2);
    Int d = c.add(carry);
    return d;
  }

  Int() : capacity(0), n_limbs(0), d_limbs(nullptr) { }

  Int& reserve(int capacity) {
    if (capacity <= this->capacity) return *this;

    this->capacity = capacity;
    if (d_limbs != nullptr)
      cudaFree(d_limbs);
    if (capacity > 0)
      cudaMalloc(&d_limbs, sizeof(Limb) * capacity);
    return *this;
  }
  
  Int& resize(int n_limbs) {
    reserve(n_limbs);
    this->n_limbs = n_limbs;
    return *this;
  }

  int capacity;
  int n_limbs; 
  Limb* d_limbs;
public:
  // TODO write test for assignment
  Int(const mpz_class& gmp_n) : capacity(0), n_limbs(0), d_limbs(nullptr) {
    //mpz_export (void *data, size_t *countp, int order,
    //	    size_t size, int endian, size_t nail, mpz_srcptr z)
    // countp is the number of bytes written
    // order : -1 for least significant limb first, 1 for most significant 1st
    // endian : 0 for native, 1 for big, -1 for little
    // size : number of bytes in a word
    // nail : not supported, 0
    // z : 
    //
    // start with a copy host side
    // use capacity and never automatically shrink
    // What about CUDA C++ interface?
    size_t n_mp_limbs = mpz_size(gmp_n.get_mpz_t());
    if (n_mp_limbs == 0) return;

    resize((int)n_mp_limbs * sizeof(mp_limb_t) / sizeof(Limb));
    // TODO remember that alignment is important for fast movement.
    // This direction probably is less important than the other direction.
    cudaMemcpy(d_limbs, (void*)mpz_limbs_read(gmp_n.get_mpz_t()), sizeof(Limb) * n_limbs, cudaMemcpyHostToDevice);
  }
  ~Int() {
    if (d_limbs != nullptr)
      cudaFree(d_limbs);
  }

  uint16_t cksum() const {
    d_Val<uint16_t> r = (uint16_t)0;
    d_cksum<<<1,1>>>(r, (uint8_t*)d_limbs, sizeof(Limb)*n_limbs);
    return r;
  }

  // an inherently large memory operation, if you need to write the number to
  // disk, then you should handle it differently.
  operator std::string() {
    if (n_limbs == 0) return "0";

    std::vector<Limb> data(n_limbs);
    cudaMemcpy(data.data(), d_limbs, sizeof(Limb) * n_limbs, cudaMemcpyDeviceToHost);

    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (auto rIt = data.rbegin(); rIt != data.rend(); rIt++) {
      if (rIt != data.rbegin()) ss << std::setw(sizeof(Limb) * 2);
      ss <<  *rIt;
    }
    ss << std::dec;
    std::string s = ss.str();
    //std::reverse(s.begin(), s.end());
    return s;
  }
  Int operator+(const Int& b) {
    return add(b);
  }
  Int operator*(const Int& b) {
    return times(b);
  }
};
}
#endif
