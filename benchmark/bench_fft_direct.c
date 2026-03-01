/*
 * bench_fft_direct.c  --  TEMPORARY
 *
 * Benchmarks CPU zz_mpnfft_poly_fft() vs GPU cu_zz_mpnfft_poly_fft() (H2D +
 * NTT) for every lgN in [GPU_MIN_THRESHOLD, GPU_MAX_THRESHOLD].  No matmul or
 * big-integer layer is involved -- we operate directly on the polynomial /
 * NTT buffers.
 *
 * Build:  make bench_fft
 * Run:    build_fft_bench/bench_fft_direct
 *
 * Columns
 *   lgN        -- transform exponent (N = 2^lgN)
 *   N          -- transform length
 *   cpu_fft    -- wall time for one forward NTT (CPU, single poly, 4 primes)
 *   gpu_fwd    -- wall time for H2D + forward NTT (GPU, includes host→device copy)
 *   gpu_rt     -- wall time for full round-trip: H2D + fwd NTT + inv NTT + D2H
 *   fwd_x      -- speedup of gpu_fwd over cpu_fft (>1 means GPU is faster)
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <gmp.h>

#include "mpzfft_moduli.h"    /* zz_moduli_t, zz_moduli_init/clear           */
#include "mpnfft.h"            /* zz_mpnfft_poly_*, cu_zz_mpnfft_poly_*       */
#include "hwmpz.h"             /* hw_mpz_setup/clear (inits global cu_zz_moduli) */
#include "cufft62_thresholds.h" /* GPU_MIN_THRESHOLD, GPU_MAX_THRESHOLD        */

/* ── configuration ─────────────────────────────────────────────────────────── */

#define NUM_PRIMES  4

/* Number of timed repetitions per size.  Fewer for large N. */
static int n_reps(int lgN)
{
    if (lgN >= 22) return 1;
    if (lgN >= 18) return 2;
    return 3;
}

/* ── helpers ────────────────────────────────────────────────────────────────── */

static double wall_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Fill a flat buffer of layout [num_primes][sz] with uniform random values
 * in [0, p[i]) for each prime i. */
static void fill_random_flat(uint64_t *buf, size_t sz,
                              unsigned num_primes, const uint64_t *p)
{
    for (unsigned i = 0; i < num_primes; i++) {
        uint64_t *row = buf + (size_t)i * sz;
        uint64_t pi   = p[i];
        for (size_t j = 0; j < sz; j++)
            row[j] = ((uint64_t)(unsigned)rand() << 17 ^ (unsigned)rand()) % pi;
    }
}

/* ── main ───────────────────────────────────────────────────────────────────── */

int main(void)
{
    srand(42);

    /* ---- initialise moduli ------------------------------------------------ */
    printf("Initialising moduli ...\n");
    fflush(stdout);

    zz_moduli_t cpu_mod;
    zz_moduli_init(&cpu_mod, NUM_PRIMES);

    /* hw_mpz_setup() sets up the global cu_zz_moduli used by
     * cu_zz_mpnfft_poly_fft internally (defined in cufftwrapper.cu). */
    hw_mpz_setup();

    printf("Done.\n\n");

    /* ---- table header ----------------------------------------------------- */
    printf("%-5s  %-10s  %-12s  %-12s  %-12s  %-8s\n",
           "lgN", "N", "cpu_fft(s)", "gpu_fwd(s)", "gpu_rt (s)", "fwd_x");
    printf("%-5s  %-10s  %-12s  %-12s  %-12s  %-8s\n",
           "---", "---------", "----------", "----------", "----------", "-------");

    /* ---- sweep lgN -------------------------------------------------------- */
    for (int lgN = GPU_MIN_THRESHOLD; lgN <= GPU_MAX_THRESHOLD; lgN++) {

        const size_t N    = (size_t)1 << lgN;
        const int    reps = n_reps(lgN);

        /* Build a params struct for this lgN.
         * Only the FFT-relevant fields are needed here (lgN, N, points,
         * num_primes, moduli).  The r/terms fields are not used by fft/ifft. */
        zz_mpnfft_params_t params;
        params.moduli     = &cpu_mod;
        params.num_primes = NUM_PRIMES;
        params.terms      = 1;
        params.r          = 62;  /* unused by FFT path */
        params.lgN        = (unsigned)lgN;
        params.N          = N;
        params.points     = N;   /* full power-of-2 transform */

        /* ================================================================
         * CPU forward FFT
         * zz_mpnfft_poly_fft(out, in, threads) reads poly_in, writes poly_out.
         * poly_in is unchanged across calls, so no re-fill needed between reps.
         * ================================================================ */
        zz_mpnfft_poly_t p_in, p_out;
        zz_mpnfft_poly_init(p_in,  &params);
        zz_mpnfft_poly_init(p_out, &params);
        zz_mpnfft_poly_alloc(p_in);
        zz_mpnfft_poly_alloc(p_out);
        p_in->size = N;  /* full input */

        /* fill p_in with values in [0, p[i]) */
        for (unsigned i = 0; i < (unsigned)NUM_PRIMES; i++) {
            uint64_t pi = cpu_mod.p[i];
            for (size_t j = 0; j < N; j++)
                p_in->data[i][j] =
                    ((uint64_t)(unsigned)rand() << 17 ^ (unsigned)rand()) % pi;
        }

        /* warmup */
        zz_mpnfft_poly_fft(p_out, p_in, /*threads=*/1);

        double cpu_total = 0.0;
        for (int r = 0; r < reps; r++) {
            double t0 = wall_sec();
            zz_mpnfft_poly_fft(p_out, p_in, 1);
            cpu_total += wall_sec() - t0;
        }
        double cpu_fft = cpu_total / reps;

        zz_mpnfft_poly_clear(p_in);
        zz_mpnfft_poly_clear(p_out);

        /* ================================================================
         * GPU forward FFT  (H2D + NTT kernel)
         * cu_zz_mpnfft_poly_fft(host_buf, params, datasz):
         *   - reduces host_buf values in-place from [0,2p) to [0,p)
         *     (no-op here since we fill with [0,p) already)
         *   - H2D copies host_buf to device
         *   - runs batched NTT on GPU
         *   - returns device pointer (caller must eventually free it)
         *
         * cu_zz_mpnfft_poly_ifft(host_dst, d_ptr, params, datasz):
         *   - runs batched INTT on GPU
         *   - D2H to host_dst
         *   - cudaFree(d_ptr)
         *
         * Layout of host buffer: [NUM_PRIMES][N]  (N uint64_t per prime)
         * ================================================================ */

        size_t bufsz    = (size_t)NUM_PRIMES * N;
        uint64_t *g_buf = malloc(bufsz * sizeof(uint64_t));
        uint64_t *g_tmp = malloc(bufsz * sizeof(uint64_t));  /* ifft output */

        if (!g_buf || !g_tmp) {
            fprintf(stderr, "  lgN=%d: malloc failed, skipping GPU.\n", lgN);
            free(g_buf);
            free(g_tmp);
            printf("%-5d  %-10zu  %-12.6f  %-12s  %-12s  %-8s\n",
                   lgN, N, cpu_fft, "OOM", "OOM", "-");
            continue;
        }

        fill_random_flat(g_buf, N, NUM_PRIMES, cpu_mod.p);

        /* GPU warmup: fwd + free via ifft */
        {
            uint64_t *d = cu_zz_mpnfft_poly_fft(g_buf, &params, (int)N);
            cu_zz_mpnfft_poly_ifft(g_tmp, d, &params, (int)N);
        }

        /* Measure just forward pass (H2D + NTT) */
        fill_random_flat(g_buf, N, NUM_PRIMES, cpu_mod.p);
        double gpu_fwd_total = 0.0;
        for (int r = 0; r < reps; r++) {
            double t0 = wall_sec();
            uint64_t *d = cu_zz_mpnfft_poly_fft(g_buf, &params, (int)N);
            gpu_fwd_total += wall_sec() - t0;
            /* free device memory outside the timer */
            cu_zz_mpnfft_poly_ifft(g_tmp, d, &params, (int)N);
        }
        double gpu_fwd = gpu_fwd_total / reps;

        /* Measure full round-trip (H2D + NTT + INTT + D2H) */
        fill_random_flat(g_buf, N, NUM_PRIMES, cpu_mod.p);
        double gpu_rt_total = 0.0;
        for (int r = 0; r < reps; r++) {
            fill_random_flat(g_buf, N, NUM_PRIMES, cpu_mod.p);
            double t0 = wall_sec();
            uint64_t *d = cu_zz_mpnfft_poly_fft(g_buf, &params, (int)N);
            cu_zz_mpnfft_poly_ifft(g_buf, d, &params, (int)N);
            gpu_rt_total += wall_sec() - t0;
        }
        double gpu_rt = gpu_rt_total / reps;

        free(g_buf);
        free(g_tmp);

        double speedup = cpu_fft / gpu_fwd;
        printf("%-5d  %-10zu  %-12.6f  %-12.6f  %-12.6f  %-8.2fx\n",
               lgN, N, cpu_fft, gpu_fwd, gpu_rt, speedup);
        fflush(stdout);
    }

    /* ---- cleanup ---------------------------------------------------------- */
    zz_moduli_clear(&cpu_mod);
    hw_mpz_clear();
    return 0;
}
