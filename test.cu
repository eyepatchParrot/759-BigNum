// Notes at bottom.
#include <iostream>
#include <string>
#include <limits>

#include <gmp.h>
#include <gmpxx.h>

#include "int.h"
#include "cksum.h"

uint16_t mpz_cksum(const mpz_class& n) {
  auto p_n = n.get_mpz_t();
  return cksum((uint8_t*)mpz_limbs_read(p_n), mpz_size(p_n) * sizeof(mp_limb_t));
}

int main() {
  using GPU::Int;

  // *** test loads and cksum
  for (int i = 0; i < 0xFFFF; i += 0x10) {
    mpz_class mpz_i = i;
    Int gpu_i(mpz_i);
    if (mpz_cksum(mpz_i) != gpu_i.cksum()) {
      std::string mp_str = mpz_i.get_str(16), d_str = gpu_i;
      if (d_str == mp_str) continue;
      std::cerr << mp_str << '\n' << d_str << '\n';
      assert(mp_str == d_str);
    }
  }
  std::cout << "PASS load inc by 16\n";

  mpz_class mpz_i = std::numeric_limits<long>::max();
  for (int i = 0; i < 100; i++, mpz_i += mpz_i) {
    Int gpu_i = mpz_i;

    if (mpz_cksum(mpz_i) != gpu_i.cksum()) {
      std::string d_str = gpu_i, mp_str = mpz_i.get_str(16);
      if (d_str == mp_str) continue;
      std::cerr << mp_str << '\n' << d_str << '\n';
      assert(mp_str == d_str);
    }
  }
  std::cout << "PASS load sum by itself\n";

  for (int i = 0; i < 10; i++, mpz_i *= mpz_i) {
    Int gpu_i = mpz_i;

    if (mpz_cksum(mpz_i) != gpu_i.cksum()) {
      std::string d_str = gpu_i, mp_str = mpz_i.get_str(16);
      if (d_str == mp_str) continue;
      std::cerr << mp_str << '\n' << d_str << '\n';
      assert(mp_str == d_str);
    }
  }

  std::cout << "PASS load squared\n";

  /*
   * test addition without carries
   */
  // single block
  mpz_class blk_n = 1L << 63;
  assert(mpz_size(blk_n.get_mpz_t()) == 1);
  blk_n *= 2;
  assert(mpz_size(blk_n.get_mpz_t()) == 2);

#define GPU_OK(N1, OP, N2) \
  assert(mpz_cksum(N1 OP N2) == (Int(N1) OP Int(N2)).cksum())

  GPU_OK(1_mpz, +, 1_mpz);
  std::cout << "PASS 1+1 = 2\n";

  GPU_OK(blk_n, +, blk_n);
  std::cout << "PASS 2**64 + 2**64\n";

  mpz_i = blk_n + 1_mpz;

  std::cout << std::string(Int(mpz_i) + Int(mpz_i)) << '\n';
  mpz_i += mpz_i;
  std::cout << mpz_i.get_str(16) << '\n';


  GPU_OK(mpz_i, +, mpz_i);
  std::cout << "PASS 2**64+1 + 2**64+1\n";

  return 0;
}
