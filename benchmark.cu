// Notes at bottom.
#include <iostream>
#include <limits>

#include <gmp.h>
#include <gmpxx.h>

#include <benchmark/benchmark.h>

#include "int.h"

void gmp_add(mpz_class& c, mpz_class& a, mpz_class& b, GPU::Int& g_c, GPU::Int& g_a, GPU::Int& g_b) {
  c = a + b;
}

void gmp_mul(mpz_class& c, mpz_class& a, mpz_class& b, GPU::Int& g_c, GPU::Int& g_a, GPU::Int& g_b) {
  c = a * b;
}

void gpu_add(mpz_class& c, mpz_class& a, mpz_class& b, GPU::Int& g_c, GPU::Int& g_a, GPU::Int& g_b) {
  g_c = g_a.add(g_b);
}

void gpu_mul(mpz_class& c, mpz_class& a, mpz_class& b, GPU::Int& g_c, GPU::Int& g_a, GPU::Int& g_b) {
  g_c = g_a.times<false>(g_b);
}

void gpu_add_s(mpz_class& c, mpz_class& a, mpz_class& b, GPU::Int& g_c, GPU::Int& g_a, GPU::Int& g_b) {
  g_c = g_a.add_serial(g_b);
}

void gpu_mul_s(mpz_class& c, mpz_class& a, mpz_class& b, GPU::Int& g_c, GPU::Int& g_a, GPU::Int& g_b) {
  g_c = g_a.times<true>(g_b);
}

using BinOp = void(mpz_class& , mpz_class& , mpz_class& , GPU::Int& , GPU::Int& , GPU::Int&);

template <BinOp *op>
static void Op(benchmark::State& state) {
  gmp_randstate_t rands;
  gmp_randinit_default(rands);
  gmp_randseed_ui(rands, 42);
  mpz_class a,b,c;
  mpz_rrandomb(a.get_mpz_t(), rands, state.range(0));
  mpz_rrandomb(b.get_mpz_t(), rands, state.range(0));
  GPU::Int g_a = a, g_b = b, g_c = c;

  for (auto _ : state) op(c, a, b, g_c, g_a, g_b);

  state.SetComplexityN(state.range(0));
}

BENCHMARK_TEMPLATE(Op, gpu_mul)->Range(4096, 2048 << (2*5));
BENCHMARK_TEMPLATE(Op, gpu_mul_s)->Range(4096, 2048 << (2*5));
BENCHMARK_TEMPLATE(Op, gmp_mul)->Range(4096, 2048 << (2*5));
//BENCHMARK_TEMPLATE(Op, true)->Range(2048, 934113382);
BENCHMARK_TEMPLATE(Op, gpu_add)->Range(2048, std::numeric_limits<int>::max());
BENCHMARK_TEMPLATE(Op, gpu_add_s)->Range(2048, std::numeric_limits<int>::max());
BENCHMARK_TEMPLATE(Op, gmp_add)->Range(2048, std::numeric_limits<int>::max());
BENCHMARK_MAIN();

// 1. GMP Add
// 2. GMP Gradeschool
//
// As far as carry is concerned, it may be useful to make operations really
// operate on promises that return immediately. so that more efficient versions
// can be used (for example, aggregating carry work across multiple carries)
//
// For testing adds, an add followed by a subtract should compare to the
// original. 
//
// Need a 'getrnd' with a limb parameter
//
// The toom22 threshold is ~33. I suspect that the two stage, highly parallel
// nature of the GPU will really change things. The algorithms used in GMP
// base case all look very serial.
//
// For base case, I can look at scheduling a base case multiplication on an
// entire number, or I can look at doing a vector of base case multiplications.
//
// toom22 : (aB + b)(cB + d) = acB^2 + bd + ((a+b)(c+d) - ac - bd)B
// So, schedule ac, bd, (a+b)(c+d) in parallel on the GPU as a vector3 mul
// kernel.
//
// Note that it might be possible that toom22 beats base case even when done
// in a single block. In that case, you could vectorize toom22.
//
// In my original design, each block was responsible for a single block of
// output by reading in up to n different inputs. This is different from the
// vector multiplication because each unit touches only its input units.
//
// There are several different kinds of units here. There's the block level unit
// but there's also the size of a unit that is scheduled on a GPU as an
// independent multiplication. With the vector scheduling, each multiplication
// is independent. We should be able to parameterize this so long as the inputs
// are roughly the same, and we say how large each unit is. Blocks just check
// which unit they're apart of and generate the limits of that unit
// independently to stay within bounds.
//
// GMP is responsible for all host side code. I am only writing device side
// code.
//
// I'll need to figure out how to go from device to host while minimizing
// memory overhead, but I'll worry about that another time.
// mpz_getlimbn will probably be useful.
//
// Here's a wild idea. What if we didn't keep an entire integer contiguous, but
// rather kept just enough contiguous to put it into a block, but also used a
// directory to allow for growth?
//
// The key question is what the performance penalty would be. The way I see it
// it would only require a single additional load from global memory, one extra
// register during each phase of loading from global to shared memory.
//
// I need some terminology to refer to a size such that an entire block works
// on it. I'll call it a block limb.
