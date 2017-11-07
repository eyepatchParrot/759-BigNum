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

namespace GPU {
__global__ void d_cksum(uint16_t* sum, const uint8_t* data, int n_bytes) {
  *sum = cksum(data, n_bytes);
}
class Int {
  using Limb = int64_t;

  template <typename T>
  class d_Val {
    T v;
    T* d_v;
  public:
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

    n_limbs = (int)n_mp_limbs * sizeof(mp_limb_t) / sizeof(Limb);
    // TODO consider including some extra space to avoid re-allocation
    // probably have a minimum size
    cudaMalloc(&d_limbs, sizeof(Limb) * n_limbs);
    // TODO remember that alignment is important for fast movement.
    // This direction probably is less important than the other direction.
    cudaMemcpy(d_limbs, (void*)mpz_limbs_read(gmp_n.get_mpz_t()), sizeof(Limb) * n_limbs, cudaMemcpyHostToDevice);
  }
  ~Int() {
    if (d_limbs != nullptr)
      cudaFree(d_limbs);
  }

  uint16_t cksum() {
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
};
}
#endif
