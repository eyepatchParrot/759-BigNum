# Warnings
#WFLAGS	:= -Wall -Wextra -Wsign-conversion -Wsign-compare
WFLAGS	:= -Wall -Wextra
CUDA_WFLAGS := $(foreach option, $(WFLAGS), --compiler-options $(option))

# Optimization and architecture
OPT		:= -O3 -DNDEBUG
ARCH   	:= -march=native

CXX := nvcc
BIN = "/usr/local/gcc/6.4.0/bin/gcc"
CXXFLAGS := -g -std=c++14
LIB := -lgmpxx -lgmp -L$(HOME)/lib -lbenchmark -lpthread
INC := -I$(HOME)/include

EXEC := test benchmark

all: $(EXEC)

.PHONY: debug
debug : OPT  := -O0 -g -G
debug : $(EXEC)

%: %.cu int.h cksum.h int.cu
	$(CXX) $(CXXFLAGS) $(OPT) $(INC) int.cu $< -o $@ $(LIB) -ccbin $(BIN)

clean:
	rm -f $(EXEC) e.* o.*

sh_test: test
	sbatch sh_test.sh

sh_bench: benchmark
	sbatch sh_bench.sh


#14: 14.cu Int.h
#	$(CXX) $(CXXFLAGS) $(INC) $< -o $@ $(LIB)
