CXX := nvcc
BIN = "/usr/local/gcc/6.4.0/bin/gcc"
CXXFLAGS := -g -std=c++14
LIB := -lgmpxx -lgmp -L$(HOME)/lib -lbenchmark -lpthread
INC := -I$(HOME)/include

EXEC := test 14

all: $(EXEC)

debug: test
	gdb ./test

%: %.cu int.h cksum.h
	$(CXX) $(CXXFLAGS) $(INC) $< -o $@ $(LIB) -ccbin $(BIN)

clean:
	rm -f $(EXEC)

sbatch: test
	sbatch slurm.sh


#14: 14.cu Int.h
#	$(CXX) $(CXXFLAGS) $(INC) $< -o $@ $(LIB)
