Goal is to write the fastest integer multiplication for acceleration on GPUs.

Build everything
$ make

Run benchmark
$ ./benchmark

Run tests
$ ./test

File main.cpp is the driver for the code, it instantiates the class objects

File _Toom_Cook_3.cpp implements the logic for split, evaluation , pointwise multiplication and interpolation

Run $gcc -std=c++11 -fopenmp main.cpp _Took_Cook__3.cpp -o accelerate

Run $accelerate

Files:
* benchmark.cu uses Google benchmark to benchmark GMP multiplication against our library.
* chksum.h is a Fletcher checksum written for use by CPU or GPU.
* int.h is the header file for a bignum on the GPU
* int.cu is the implementation file.
* slurm.sh is a SLURM script for running tests on the GPU
* test.cu is a test suite for our library. Some inspiration from GMP.
