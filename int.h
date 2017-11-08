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
 */

namespace GPU {
  using Limb = int64_t;

__global__ void d_cksum(uint16_t* sum, const uint8_t* data, int n_bytes) {
  *sum = cksum(data, n_bytes);
}
__global__ void d_add(Limb* c, int c_sz, const Limb* a, const Limb* b, int b_sz, Limb* carry) {
  // each block needs a carry, so you need an array of them.
  // can re-use b if it's mutable. If not, make a copy.
  // each thread gets its own limb of the output.
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < b_sz)
    c[tid] = a[tid] + b[tid];
  else if (tid < c_sz)
    c[tid] = a[tid];
  __syncthreads();

  if (threadIdx.x == 0) {
    int tCarry = 0;
    for (int i = 0; i < blockDim.x && tid+i < c_sz; i++) {
      // need to make sure that you handle the case where the carry causes
      // the carry and the case where you maxed out the carry and had a carry
      // 0xffff + 0xffff + 1 = 0x1fffe + 1 = 0x1ffff
      c[tid+i] += tCarry;
      tCarry = tCarry ? c[tid+i] <= a[tid+i] : c[tid+i] < a[tid+i];
    }
    carry[blockIdx.x] = tCarry;
  }
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

    Int c = std::move(Int().resize(a.n_limbs));

    // assumes fewer than 1024 * add_block_sz limbs
    d_add<<<n_blocks, add_block_sz>>>(c.d_limbs, c.n_limbs, a.d_limbs, b.d_limbs, b.n_limbs, d_carry);
    // TODO figure out carries
//    int block_sz = add_block_sz;
//    do {
//      n_blocks = (n_blocks - 1 + carry_block_sz) / carry_block_sz;
//      d_carry<<<n_blocks, carry_block_sz>>>(c.d_limbs, c.n_limbs, d_carry, block_sz);
//      block_sz *= carry_block_sz;
//    } while (n_blocks > 1);

    cudaFree(d_carry);
    return c;
  }

  Int() : n_limbs(0), d_limbs(nullptr) { }
  //Int(Int&& b) : n_limbs(b.n_limbs), d_limbs(b.d_limbs) {
  //  b.n_limbs = 0;
  //  b.d_limbs = nullptr;
  //}
  //Int& operator=(Int&& b) {
  //  if (d_limbs != nullptr)
  //    cudaFree(d_limbs);
  //  d_limbs = b.d_limbs;
  //  n_limbs = b.n_limbs;
  //  b.d_limbs = nullptr;
  //  b.n_limbs = 0;
  //  return *this;
  //}

  Int& resize(int n_limbs) {
    this->n_limbs = n_limbs;
    if (d_limbs != nullptr)
      cudaFree(d_limbs);
    if (n_limbs > 0)
      cudaMalloc(&d_limbs, sizeof(Limb) * n_limbs);
    return *this;
  }

  int n_limbs; 
  Limb* d_limbs;
public:
  // TODO write test for assignment
  Int(const mpz_class& gmp_n) : n_limbs(0), d_limbs(nullptr) {
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
    d_Val<uint16_t> r = 0;
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
