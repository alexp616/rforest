# mpz_matmul

This repository takes the matrix multiplication step, implemented in `mpz_rmatrix_mult_fft()`, from David Harvey and Andrew Sutherland's [rforest](https://math.mit.edu/~drew/rforest.html) and attempts to optimize it via GPU parallelization. The end goal is to have a drop in replacement for their matrix multiplication function to be used in rforest, as well as other applications.

Dependencies: [GMP](https://gmplib.org/), [CMake](https://cmake.org/download/)>=3.2, [GCC](https://gcc.gnu.org/), [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

Arbitrary-width integer matrix multiplication takes the following steps:
1. Convert integer entries to polynomial form
2. FFT polynomial form entries modulo multiple primes
3. Do matrix multiplication in frequency domain
4. IFFT frequency domain entries
5. CRT polynomials
6. Recompose (or, evaluate) polynomial to get integer

As of now, steps 2-5 are parallelized with the help of [GPU-NTT](https://github.com/Alisah-Ozcan/GPU-NTT). Step 6 should be parallelizable via some sort of segmented prefix scan, and step 1 should be parallelizable once I learn exactly how it works.

## Testing and Benchmarking

A makefile that calls CMake is provided. To test and benchmark, just run:

```bash
$ make test
$ make bench
```
