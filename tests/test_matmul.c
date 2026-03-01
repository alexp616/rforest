/*
  Test that mpz_rmatrix_mult_fft produces the same result as
  mpz_rmatrix_mult_mod_naive (the latter being a trivial O(rd^2)
  implementation used as a ground-truth reference).

  We compare C_fft  mod m  ==  C_naive  (which is already reduced mod m).

  Usage:  ./test_matmul [bits] [d] [r]
    bits  = approximate bit-size of matrix entries  (default 1000)
    d     = matrix dimension                        (default 3)
    r     = number of row-blocks in A               (default 2)
*/
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <gmp.h>
#include "hwmpz.h"

int main(int argc, char *argv[])
{
    long bits = 1000;
    int  d    = 3;
    int  r    = 2;

    if (argc > 1) bits = atol(argv[1]);
    if (argc > 2) d    = atoi(argv[2]);
    if (argc > 3) r    = atoi(argv[3]);

    printf("Testing mpz_rmatrix_mult_fft vs mpz_rmatrix_mult_mod_naive\n");
    printf("  bits = %ld,  d = %d,  r = %d\n", bits, d, r);

    gmp_randstate_t rng;
    gmp_randinit_default(rng);
    gmp_randseed_ui(rng, (unsigned long)time(NULL));

    /* --- Initialise hw memory / fft subsystem --- */
    hw_mem_init(0);
    hw_mpz_setup();

    /* --- Allocate matrices --- */
    mpz_t *A     = mpz_vec_alloc_and_init(r * d);   /* r x d */
    mpz_t *B     = mpz_vec_alloc_and_init(d * d);   /* d x d */
    mpz_t *C_fft = mpz_vec_alloc_and_init(r * d);   /* result from fft */
    mpz_t *C_ref = mpz_vec_alloc_and_init(r * d);   /* result from naive */

    /* --- Fill with random data --- */
    for (int i = 0; i < r * d; i++)
        mpz_urandomb(A[i], rng, bits);
    for (int i = 0; i < d * d; i++)
        mpz_urandomb(B[i], rng, bits);

    /* --- Work variable --- */
    mpz_t w;
    mpz_init(w);

    /* --- Compute reference: C_ref = A*B (naive, exact) --- */
    mpz_rmatrix_mult_naive(C_ref, A, r, B, d, w);

    /* --- Compute: C_fft = A*B  (fft, exact) --- */
    mpz_rmatrix_mult_fft(C_fft, A, r, B, d, w);

    /* --- Compare --- */
    int ok = 1;
    for (int i = 0; i < r * d; i++) {
        if (mpz_cmp(C_fft[i], C_ref[i]) != 0) {
            printf("MISMATCH at index %d  (row %d, col %d)\n",
                   i, i / d, i % d);
            ok = 0;
        }
    }

    if (ok)
        printf("PASS — all %d entries match.\n", r * d);
    else
        printf("FAIL\n");

    /* --- Cleanup --- */
    mpz_clear(w);
    mpz_vec_clear_and_free(A, r * d);
    mpz_vec_clear_and_free(B, d * d);
    mpz_vec_clear_and_free(C_fft, r * d);
    mpz_vec_clear_and_free(C_ref, r * d);

    hw_mpz_clear();
    hw_mem_clear();

    gmp_randclear(rng);

    return ok ? 0 : 1;
}
