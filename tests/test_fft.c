#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gmp.h>
#include <fft62.h>

typedef struct cu_fft62_mod_t cu_fft62_mod_t;
extern cu_fft62_mod_t cu_mod;
extern void cu_fft62_mod_init(cu_fft62_mod_t* mod, uint64_t p);
extern void cu_fft62_fft(uint64_t* yp, uint64_t* xp, size_t size, unsigned lgN, cu_fft62_mod_t* mod);
extern void cu_fft62_ifft(uint64_t* yp, uint64_t* xp, unsigned lgN, cu_fft62_mod_t* mod);
extern void cu_fft62_mod_clear(cu_fft62_mod_t* mod);

void print_arr(uint64_t* arr, int n) {
    printf("[");
    for (int i = 0; i < n - 1; ++i) {
        printf("%lu, ", arr[i]);
    }
    printf("%lu]\n", arr[n - 1]);
    return;
}

void fft_test1() {
    // Initialize mod
    const uint64_t p = 0x3fffc00000000001ULL;

    // FFT size
    const int LGN = 20;
    const int N = 1 << LGN;
    const size_t yn = N;
    const size_t xn = N;

    // Init vectors
    uint64_t* xp = malloc(xn * sizeof(uint64_t));
    uint64_t* yp = malloc(yn * sizeof(uint64_t));
    uint64_t* zp = malloc(xn * sizeof(uint64_t));

    for (int i = 0; i < xn; ++i) {
        xp[i] = 1;
    }

    // Initialize mod for gpu
    // printf("Creating cu_fft62_mod_t\n");
    cu_fft62_mod_init(&cu_mod, p);
    // printf("Doing FFT\n");
    cu_fft62_fft(yp, xp, 1 << LGN, LGN, &cu_mod);
    // printf("\n");
    // printf("Finishing FFT\n");
    // printf("Doing inverse FFT\n");
    cu_fft62_ifft(zp, yp, LGN, &cu_mod);

    for (int i = 0; i < xn; ++i) {
        assert(zp[i] == N);
    }

    free(xp);
    free(yp);
    free(zp);
    cu_fft62_mod_clear(&cu_mod);
    return;
}

void fft_test2() {
    const uint64_t p = 0x3fffc00000000001ULL;
    fft62_mod_t mod = {};
    fft62_mod_init(&mod, p);
    cu_fft62_mod_init(&cu_mod, p);
    for (int LGN = 11; LGN <= 20; ++LGN) {
        int N = 1 << LGN;
        size_t xn = N;

        uint64_t* xp = malloc(xn * sizeof(uint64_t));
        uint64_t* yp = malloc(xn * sizeof(uint64_t));
        uint64_t* zp = malloc(xn * sizeof(uint64_t));

        for (int i = 0; i < xn; ++i) { xp[i] = rand() % 12345; }

        fft62_fft(yp, xn, xp, xn, LGN, &mod, 1);
        cu_fft62_fft(zp, xp, 1 << LGN, LGN, &cu_mod);

        fft62_ifft(yp, xn, yp, LGN, &mod, 1);
        cu_fft62_ifft(zp, zp, LGN, &cu_mod);

        for (int i = 0; i < xn; ++i) { assert((yp[i] % p) == zp[i]); }
        
        free(xp);
        free(yp);
        free(zp);
        
    }
    fft62_mod_clear(&mod);
    cu_fft62_mod_clear(&cu_mod);
    
    return;
}

int main (int argc, char *argv[])
{
    fft_test1();
    fft_test2();

    printf("All cufft tests passed!\n");

    return 0;
}
