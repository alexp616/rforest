# rforest

This repository is a fork of David Harvey and Andrew Sutherland's [rforest](https://math.mit.edu/~drew/rforest.html). We plan to accelerate the algorithm by implementing parts of it (and eventually maybe the entire thing) in CUDA. Primarily, we aim to optimize the main bottleneck, which is the matrix multiplication function, `mpz_rmatrix_mult_fft()`, implemented in `src/hwmpz.c`

Dependencies: [GMP](https://gmplib.org/), [CMake](https://cmake.org/download/)>=3.2, [GCC](https://gcc.gnu.org/), [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

## Testing and Benchmarking

A makefile that calls CMake is provided. To test and benchmark, just run:

```bash
$ make test
$ make bench
```