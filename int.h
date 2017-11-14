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
  using Limb = uint64_t;

__global__ void d_cksum(uint16_t* sum, const uint8_t* data, int n_bytes) {
  *sum = cksum(data, n_bytes);
}
__global__ void g_add(Limb* c,  const Limb* a, const int a_sz, const Limb* b, int b_sz, Limb* carry) {
  // each block needs a carry, so you need an array of them.
  // can re-use b if it's mutable. If not, make a copy.
  // each thread gets its own limb of the output.
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < b_sz)
    c[tid] = a[tid] + b[tid];
  else if (tid < a_sz)
    c[tid] = a[tid];
  __syncthreads();

  if (threadIdx.x == 0) {
    int tCarry = 0;
    for (int i = 0; i < blockDim.x; i++) {
      if ((tid+i) == a_sz) {
        c[tid+i] = tCarry;
        tCarry = 0;
        break;
      }

      // need to make sure that you handle the case where the carry causes
      // the carry and the case where you maxed out the carry and had a carry
      // 0xffff + 0xffff + 1 = 0x1fffe + 1 = 0x1ffff
      c[tid+i] = c[tid+i] + tCarry;
      tCarry = tCarry ? c[tid+i] <= a[tid+i] : c[tid+i] < a[tid+i];
    }
    carry[blockIdx.x] = tCarry;
  }
}

template <typename T>
T d_read(T* d) {
  T t;
  cudaMemcpy(&t, d, sizeof(T), cudaMemcpyDeviceToHost);
  return t;
}

__global__ void g_carry(Limb* c, int n_c, Limb* carry, int block_sz) {
  if (threadIdx.x != 0) return;

  int t_carry = 0;
  for (int i = block_sz; i < n_c; i++) {
    if (i % block_sz == 0) t_carry += carry[(i / block_sz) - 1];
    Limb old_c = c[i];
    c[i] += t_carry;
    t_carry = c[i] < old_c ? 1 : 0;
  }
  if (n_c % block_sz != 0) return;
  int n_blocks = (n_c-1 + block_sz) / block_sz;
  // assert n_c is valid and no overflow
  // only include carry[n_blocks-1] if n_c falls on a block boundary. Otherwise, it should've been simply included.
  c[n_c] = t_carry + carry[n_blocks-1];
}

class Int {
  static constexpr int add_block_sz = 256;
  static constexpr int carry_block_sz = 256;
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

  Int add(const Int& other) {
    // I need some way to not allocate and free the carry every time.
    const Int& a = this->n_limbs >= other.n_limbs ? *this : other;
    const Int& b = this->n_limbs >= other.n_limbs ? other : *this;

    int n_blocks = (a.n_limbs - 1 + add_block_sz) / add_block_sz;
    Limb* d_carry;
    cudaMalloc(&d_carry, sizeof(Limb) * n_blocks);

    Int c;
    c.reserve(a.n_limbs + 1);
    c.resize(a.n_limbs);
    //Int c = std::move(Int().reserve(a.n_limbs));

    // assumes fewer than 1024 * add_block_sz limbs
    g_add<<<n_blocks, add_block_sz>>>(c.d_limbs, a.d_limbs, a.n_limbs, b.d_limbs, b.n_limbs, d_carry);
    // n_limbs vs capacity
    assert(c.capacity > c.n_limbs);
    g_carry<<<1, 1>>>(c.d_limbs, c.n_limbs, d_carry, add_block_sz);
    if (d_read(c.d_limbs + c.n_limbs) != 0) c.n_limbs++;

    // This is an iterative approach. Get it working first.
    //int block_sz = add_block_sz;
    //do {
    //  n_blocks = (n_blocks - 1 + carry_block_sz) / carry_block_sz;
    //  d_carry<<<n_blocks, carry_block_sz>>>(c.d_limbs, c.n_limbs, d_carry, block_sz);
    //  block_sz *= carry_block_sz;
    //} while (n_blocks > 1);

    cudaFree(d_carry);
    return c;
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
};
}
#endif
