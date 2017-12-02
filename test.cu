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
  // single limb
  mpz_class limb_n = 1_mpz << 64;
  assert(mpz_size(limb_n.get_mpz_t()) == 2);


#define GPU_OK(N1, OP, N2, NAME) \
  do { \
    mpz_class t1 = N1, t2 = N2; \
    Int t3 = Int(t1) OP Int(t2); \
    if (mpz_cksum(t1 OP t2) != t3.cksum()) { \
      mpz_class r = t1 OP t2; \
        std::cerr << r.get_str(16) << '\n' << std::string(t3) << '\n'; \
        assert(!NAME); \
    } else { \
      std::cout << "PASS " << NAME << '\n'; \
    } \
  } while (0)

  GPU_OK(1_mpz, +, 1_mpz, "1+1=2");
  GPU_OK(limb_n, +, limb_n, "10 + 10 = 20");
  GPU_OK(limb_n + 1_mpz, +, limb_n + 1_mpz, "11 + 11 = 22");
  GPU_OK((limb_n - 1) * limb_n, +, (limb_n - 1), "F0 + 0F = FF");

  // carry tests
  GPU_OK(limb_n - 1, +, limb_n - 1, "F + F = 1E");
  GPU_OK(limb_n - 1, +, 1, "F + 1 = 10");
  GPU_OK((limb_n - 1) * limb_n, +, (limb_n - 1) * limb_n, "F0 + F0 = 1E0");
  GPU_OK((limb_n * limb_n - 1), +, (limb_n * limb_n - 1), "FF + FF = 1FE");
  GPU_OK((limb_n * limb_n - 1), +, 1_mpz, "FF + 1 = 100");
  GPU_OK(limb_n * (limb_n - 2) + limb_n - 1, +, limb_n + 1, "EF + 11 = 100");

  // carry block tests
  mpz_class block_n = 1_mpz << (64*256);
  assert(mpz_size(block_n.get_mpz_t()) == 257);
  GPU_OK(block_n, +, block_n, "BLK 10 + 10 = 20");
  GPU_OK(block_n - 1, +, block_n - 1, "BLK F + F = 1E");
  GPU_OK(block_n - 1, +, 1, "BLK F + 1 = 10");
  GPU_OK((block_n - 1) * block_n, +, (block_n - 1) * block_n, "BLK F0 + F0 = 1E0");
  GPU_OK((block_n * block_n - 1), +, (block_n * block_n - 1), "BLK FF + FF = 1FE");
  GPU_OK((block_n * block_n - 1), +, 1_mpz, "BLK FF + 1 = 100");
  GPU_OK(block_n * (block_n - 2) + block_n - 1, +, block_n + 1, "BLK EF + 11 = 100");

  // times tests
  GPU_OK(1_mpz, *, 1_mpz, "1 * 1 = 1");
  GPU_OK(limb_n, *, 1_mpz, "10 * 1 = 10");
  GPU_OK(block_n, *, 1_mpz, "BLK 10 * 1 = 10");
  GPU_OK(limb_n - 1, *, 2_mpz, "F * 2 = 1E");
  GPU_OK((limb_n - 1) * limb_n, *, 2_mpz, "F0 * 2 = 1E0");
  GPU_OK((limb_n * limb_n - 1), *, 2_mpz, "FF * 2 = 1FE");
  GPU_OK(block_n, *, 2_mpz, "BLK 10 * 2 = 20");
  GPU_OK(block_n - 1, *, 2_mpz, "BLK F * 2 = 1E");
  GPU_OK(limb_n - 1, *, limb_n - 1, "F * F = FFFE0001");
  GPU_OK(block_n - 1, *, block_n - 1, "BLK F * F = FFFE0001");

  // TODO randomized testing


  return 0;
}
