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

__device__ int tid() { return blockIdx.x * blockDim.x + threadIdx.x; }

__global__ void g_add(Limb* c,  const Limb* a, const int a_sz, const Limb* b, int b_sz, Limb* carry) {
  // each block needs a carry, so you need an array of them.
  // can re-use b if it's mutable. If not, make a copy.
  // each thread gets its own limb of the output.

  if (tid() < b_sz)
    c[tid()] = a[tid()] + b[tid()];
  else if (tid() < a_sz)
    c[tid()] = a[tid()];
  __syncthreads();

  if (threadIdx.x == 0) {
    int tCarry = 0;
    for (int i = 0; i < blockDim.x; i++) {
      if ((tid()+i) == a_sz) {
        c[tid()+i] = tCarry;
        tCarry = 0;
        break;
      }

      // need to make sure that you handle the case where the carry causes
      // the carry and the case where you maxed out the carry and had a carry
      // 0xffff + 0xffff + 1 = 0x1fffe + 1 = 0x1ffff
      c[tid()+i] = c[tid()+i] + tCarry;
      tCarry = tCarry ? c[tid()+i] <= a[tid()+i] : c[tid()+i] < a[tid()+i];
    }
    carry[blockIdx.x] = tCarry;
  }
}

__global__ void g_times(Limb* c, const Limb* a, const int a_n, const Limb* b, int b_n, Limb* carry) {
  // c_n is really only enough space for the low bits of the most significant limb. How much overflow is possible? 2 limbs. One for multiplication, one for sums < 2^32 ops.
  const int c_n = a_n + b_n - 1;

  // how to handle carries? multiplication has sizeof(Limb) carries
  // If you do the addition inline, you can reduce that to 1 bit per limb.
  // It might be good to do this in a warp-aware manner.
  // O(n) operands per output, so need O(lg N) additional space for carries.
  // Could reduce carry to block level, but easier to just use addition logic.
  //
  // The reason for step 6 is so that the serial addition takes only two operands.
  // 1. Multiply ops together.
  // 2. Sum low bits and carry into high bits.
  // 3. Sum high bits and carry bits.
  // 4. Repeat 1-3 for all ops.
  // 5. Save lo into c.
  // 6. Sync and add high to lo[i+1], saving sum of carry and existing carry somewhere.
  // 7. Sync and serially reduce carry to block-level.
  // 8. Save block-level carries.
  // assumes carry won't overflow which is true for min(a_n + 1, b_n + 1) < 2^32
  Limb lo = 0, hi = 0, hi2 = 0;
  int c_i = tid();
  if (c_i >= c_n) return;
  for (int a_i = min(a_n-1, c_i), b_i; (b_i = c_i - a_i) < b_n; a_i--) {
    Limb old_lo = lo, old_hi = hi;
    lo += a[a_i] * b[b_i];
    hi += __umulhi(a[a_i], b[b_i]);
    if (lo < old_lo) {
      if (++hi <= old_hi) hi2++;
    } else if (hi < old_hi) hi2++;
  }
  c[c_i] = lo;
  if (threadIdx.x < 2) carry[c_i] = 0;
  __syncthreads();

  // remember, c != carry. carry has the extra room, but c doesn't.
  if (threadIdx.x < blockDim.x - 1 && (c_i+1 < c_n)) {
    Limb old_lo = c[c_i+1];
    c[c_i+1] += hi;
    if (c[c_i+1] < old_lo) hi2++;
    carry[c_i+2] = hi2;
  }
  __syncthreads();
  
  if (threadIdx.x == blockDim.x - 1 || !(c_i+1 < c_n)) {
    Limb old_lo = carry[c_i+1];
    carry[c_i+1] += hi;
    if (carry[c_i+1] < old_lo) hi2++;
    carry[c_i+2] = hi2;
  }

  // Note that at the block level, you may have two limbs of carry. As far as a
  // ripple carry is concerned. The sum with the lower limb generates a single
  // bit for the upper limb, which overflows only if it is the maximum value
  // which is true only if min(a_n + 2, b_n + 2) >= 2^32.
  // This can be handled in g_carry if a carry size is used with a while (i % m < z).
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
  static constexpr int times_block_sz = 256;
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

    Int c, carry;
    c.resize(a.n_limbs + 1);
    carry.resize(n_blocks);
    //Int c = std::move(Int().reserve(a.n_limbs));

    // assumes fewer than 1024 * add_block_sz limbs
    g_add<<<n_blocks, add_block_sz>>>(c.d_limbs, a.d_limbs, a.n_limbs, b.d_limbs, b.n_limbs, carry.d_limbs);
    g_carry<<<1, 1>>>(c.d_limbs, a.n_limbs, carry.d_limbs, add_block_sz);
    if (d_read(c.d_limbs + a.n_limbs) == 0) c.resize(a.n_limbs);

    // This is an iterative approach. Get it working first.
    //int block_sz = add_block_sz;
    //do {
    //  n_blocks = (n_blocks - 1 + carry_block_sz) / carry_block_sz;
    //  d_carry<<<n_blocks, carry_block_sz>>>(c.d_limbs, c.n_limbs, d_carry, block_sz);
    //  block_sz *= carry_block_sz;
    //} while (n_blocks > 1);
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
