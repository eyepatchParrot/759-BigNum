#include "int.h"

__device__ void s_reduce(volatile Limb* p, volatile Limb* g) {
  // [[2 * k * (1+i) - 1 for i in range(16) if (2 * k * (1+i)-1) < 16]
  //  for k in [2**j for j in range(4)]]
  for (int k = 1; 2 * k <= Int::add_block_sz; k *= 2) {
    int j = 2 * k * (1+threadIdx.x) - 1;
    __syncthreads();
    if (j < Int::add_block_sz) {
      Limb pj = p[j];
      p[j] = pj & p[j-k];
      g[j] = g[j] | (g[i] & pj);
    }
  }
}

__global__ void g_carry_reduce_limb_block(Limb* p, Limb* g, const Limb* a,
    const int a_sz, const Limb* b, const int b_sz) {
  __shared__ volatile Limb s_p[Int::add_block_sz];
  __shared__ volatile Limb s_g[Int::add_block_sz];

  // TODO Consider doing n operations in registers to reduce shared memory pressure.
  // Calculate limb-wise carry / propogate. Likely better than doing it bitwise.
  if (tid() < b_sz)
    s_p[threadIdx.x] = a[tid()] + b[tid()] + 1 == 0;
  else if (tid() < a_sz)
    s_p[threadIdx.x] = a[tid()] + 1 == 0;
  else // could end up doing additional work, but simplifies.
    s_p[threadIdx.x] = 0;
   
  if (tid() < b_sz)
    s_g[threadIdx.x] = (a[tid()] + b[tid()]) < a[tid()];
  else
    s_g[threadIdx.x] = 0; // x + 0 < x = false

  s_reduce(s_g, s_p);
  __syncthreads();

  // only need one output since no re-using limb propagate + generate
  if (threadIdx.x == blockDim.x - 1) {
    // no race condition since first time p and g are used
    p[blockIdx.x] = s_p[threadIdx.x];
    g[blockIdx.x] = s_g[threadIdx.x];
  }
}

__global__ void g_reduce(Limb* p, Limb* g, const int n, const int block_sz) {
  __shared__ volatile Limb s_p[Int::add_block_sz];
  __shared__ volatile Limb s_g[Int::add_block_sz];
  int k = block_sz;
  int j = (1 + tid()) * k - 1;
  // [[(1+i)*k - 1 for i in range(16) if (1+i)*k <= n]
  //  for k in [2**j for j in range(10)] if k <= n]
  if (j < n) {
    s_p[threadIdx.x] = p[j];
    s_g[threadIdx.x] = g[j];
  } else {
    s_p[threadIdx.x] = 0;
    s_g[threadIdx.x] = 0;
  }
  s_reduce(s_g, s_p);
  __syncthreads();

  if (j >= n) return;
  p[j] = s_p[threadIdx.x];
  g[j] = s_g[threadIdx.x];
}

__device__ void s_sweep(volatile Limb *p, volatile Limb *g) {
  for (unsigned k = Int::add_block_sz / 4; k >= 1; k /= 2) {
    __syncthreads();
    // [[(2 * t + 3) * B - 1 for t in range(0,16) if (2 * t + 3) * B - 1 < 16]
    //  for B in [2**j for j in range(2,-1,-1)]]
    int j = (2 * threadIdx.x + 3) * k - 1;
    if (j < blockDim.x) {
      Limb pj = p[j];
      p[j] = pj & p[j-k];
      g[j] = g[j] | (g[i] & pj);
    }
  }
}

__global__ void g_sweep(Limb* p, Limb* g, const int n, const int block_sz) {
  __shared__ volatile Limb s_p[Int::add_block_sz];
  __shared__ volatile Limb s_g[Int::add_block_sz];

  // assume that k is the smallest distance considered
  int k = block_sz;
  int i = (tid()+2) * block_sz - 1;
  // [[t * B - 1 for t in range(2,16) if (t+1)*B-1 < 16]
  //  for B in [2**j for j in range(2,-1,-1)]]
  if (i + block_sz < n) {
    s_p[threadIdx.x] = p[i];
    s_g[threadIdx.x] = g[i];
  } else
    s_p[threadIdx.x] = s_g[threadIdx.x] = 0;
  s_sweep(s_g, s_p);
  __syncthreads();
  if (i + block_sz >= n) return;
  // TODO the last propagate can be elided during addition since the carry is zero.
  p[i] = s_p[threadIdx.x];
  g[i] = s_g[threadIdx.x];
}

/*
 * If we templatize based on add / sub, can simply do invert and carry inline without additional op. However, requires a > b.
 */
// TODO merge last add into last sweep
// TODO merge last reduce into first add.
__global__ void g_add(Limb* c,  const Limb* a, const int a_sz, const Limb* b, const int b_sz, const Limb* p, const Limb *g) {
  // p and g are in terms of blocks, but to verify carry math, use serial add.
  // TODO use a parallel prefix adder to get log time addition. probably can re-use existing fns
  if (threadIdx.x != 0) return;
  if (blockIdx.x == blockDim.x - 1) c[a_sz] = g[blockIdx.x];
  int t_carry = blockIdx.x - 1 < 0 ? 0 : g[blockIdx.x - 1];
  for (int i = 0; i < blockDim.x && tid() + i < a_sz; i++) {
    int old_a = a[tid() + i];
    c[tid()+i] = old_a + b[tid() + i] + t_carry;
    t_carry = t_carry ? c[tid() + i] <= old_a : c[tid() + i] < a[tid() + i];
  }
  //if (tid() < b_sz) {
  //  Limb p_i = a[tid()] ^ b[tid()];
  //  c[tid()] = p_i ^ g[tid()];
  //} else if (tid() < a_sz) {
  //  c[tid()] = a[tid()] ^ g[tid()];
  //} else if (tid() == a_sz) {
  //  c[tid()] = g[tid()];
  //}
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

__global__ void g_carry_serial(Limb* c, int n_c, Limb* carry, int block_sz) {
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

__global__ void g_add_serial(Limb* c,  const Limb* a, const int a_sz, const Limb* b, int b_sz, Limb* carry) {
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

