#ifndef CKSUM_H
#define CKSUM_H

static __device__ __host__ uint16_t cksum(const uint8_t *data, int n_bytes) {
  uint16_t hi = 0, lo = 0;
  for (int i = 0; i < n_bytes; i++) {
    lo = (lo + data[i]) % 255;
    hi = (hi + lo) % 255;
  }
  return (hi << 8) | lo;
}

#endif
