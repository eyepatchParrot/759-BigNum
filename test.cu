// Notes at bottom.
#include <iostream>
#include <string>
#include <limits>

#include <gmp.h>
#include <gmpxx.h>

#include "int.h"
#include "cksum.h"

int main() {
  using GPU::Int;

  // *** test loads and cksum
  for (int i = 0; i < 0xFFFF; i += 0x10) {
    mpz_class mpz_i = i;
    Int gpu_i = mpz_i;
    auto mpz_t_i = mpz_i.get_mpz_t();
    auto mpz_ck = cksum((uint8_t*)mpz_limbs_read(mpz_t_i), mpz_size(mpz_t_i) * sizeof(mp_limb_t)),
         gpu_ck = gpu_i.cksum();

    if (mpz_ck != gpu_ck) {
      std::string mp_str = mpz_i.get_str(16), d_str = gpu_i;
      if (d_str == mp_str) continue;
      std::cerr << mp_str << '\n' << d_str << '\n';
      assert(mp_str == d_str);
    }
  }
  mpz_class mpz_i = std::numeric_limits<long>::max();
  for (int i = 0; i < 100; i++, mpz_i += mpz_i) {
    Int gpu_i = mpz_i;
    auto mpz_t_i = mpz_i.get_mpz_t();
    auto mpz_ck = cksum((uint8_t*)mpz_limbs_read(mpz_t_i), mpz_size(mpz_t_i) * sizeof(mp_limb_t)),
         gpu_ck = gpu_i.cksum();

    if (mpz_ck != gpu_ck) {
      std::string d_str = gpu_i, mp_str = mpz_i.get_str(16);
      if (d_str == mp_str) continue;
      std::cerr << mp_str << '\n' << d_str << '\n';
      assert(mp_str == d_str);
    }
  }

  for (int i = 0; i < 10; i++, mpz_i *= mpz_i) {
    Int gpu_i = mpz_i;
    auto mpz_t_i = mpz_i.get_mpz_t();
    auto mpz_ck = cksum((uint8_t*)mpz_limbs_read(mpz_t_i), mpz_size(mpz_t_i) * sizeof(mp_limb_t)),
         gpu_ck = gpu_i.cksum();

    if (mpz_ck != gpu_ck) {
      std::string d_str = gpu_i, mp_str = mpz_i.get_str(16);
      if (d_str == mp_str) continue;
      std::cerr << mp_str << '\n' << d_str << '\n';
      assert(mp_str == d_str);
    }
  }

  return 0;
}
