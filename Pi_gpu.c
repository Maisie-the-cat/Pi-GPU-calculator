/*
 * pi_gpu.c
 *
 * Computes as many digits of Pi as will fit in available RAM,
 * parallelizing the heavy multi-precision arithmetic across the
 * AMD RDNA 2 GPU via OpenMP + OpenCL.
 *
 * Designed for the Steam Deck APU (AMD Aerith, RDNA 2, 16 GB unified).
 *
 * Dependencies:
 *   - OpenCL (libOpenCL)
 *   - OpenMP
 *   - GMP (libgmp)
 *
 * Build (SteamOS / Linux):
 *   gcc -fopenmp -O2 -o pi_gpu pi_gpu.c -lOpenCL -lgmp -lm
 *
 * Run:
 *   OMP_NUM_THREADS=4 ./pi_gpu
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#include <CL/cl.h>
#include <gmp.h>

/* ─── Configuration ──────────────────────────────────────────── */

#define SERIAL_CUTOFF    512       /* below this, compute on CPU   */
#define GPU_BATCH_SIZE   4096      /* terms per GPU dispatch      */
#define MAX_DIGITS_CAP   500000000LL /* 500M digit safety cap     */
#define MEM_BUDGET_FRAC  0.40      /* fraction of RAM to use      */

/* Chudnovsky constants */
#define CHD_A            13591409LL
#define CHD_B            545140134LL
#define CHD_C3_OVER_24   10939058860032000LL
#define CHD_D            426880LL
#define CHD_E            10005LL
#define DIGITS_PER_TERM  14.181647462725477

/* ─── OpenCL kernel source ───────────────────────────────────── */

static const char *kernel_source = R"(

/* Each big integer is stored as an array of 32-bit limbs
   in little-endian order (limb[0] = least significant).
   We use 32-bit limbs so that a 64-bit product of two limbs
   fits in a long without overflow. */

#define LIMB_BITS 32
#define LIMB_MASK 0xFFFFFFFFUL

/* ── Kernel 1: big integer multiply (schoolbook) ───────────────
   Computes: result[0..na+nb-2] = a[0..na-1] * b[0..nb-1]
   Each work-item computes one output limb. */

__kernel void big_mul(
    __global const uint *a, int na,
    __global const uint *b, int nb,
    __global uint *result)
{
    int gid = get_global_id(0);
    int out_len = na + nb - 1;
    if (gid >= out_len) return;

    ulong sum = 0;
    int i_start = max(0, gid - nb + 1);
    int i_end   = min(gid, na - 1);

    for (int i = i_start; i <= i_end; i++) {
        sum += (ulong)a[i] * (ulong)b[gid - i];
    }
    result[gid] = (uint)(sum & LIMB_MASK);
}

/* ── Kernel 2: carry propagation ───────────────────────────────
   Normalizes a big integer in-place: propagates carries so that
   each limb fits in 32 bits. */

__kernel void propagate_carry(
    __global ulong *data,   /* input: raw products (64-bit each) */
    __global uint *out,     /* output: normalized 32-bit limbs  */
    int n)
{
    int gid = get_global_id(0);
    if (gid >= n) return;

    ulong val = data[gid];
    /* We do a single-pass carry; the host iterates this if needed. */
    out[gid] = (uint)(val & LIMB_MASK);
    if (gid + 1 < n) {
        data[gid + 1] += (val >> LIMB_BITS);
    }
}

/* ── Kernel 3: leaf term evaluation ────────────────────────────
   Each work-item computes P(k), Q(k), T(k) for one value of k.
   Results are stored as flat arrays of 32-bit limbs.

   P(k) = (6k-5)(6k-1)(2k-1)
   Q(k) = (C^3/24) * k^3
   T(k) = (A + B*k) * P(k) * (-1)^k
*/

__kernel void eval_leaf_terms(
    const long first_k,
    const int num_terms,
    const int nlimbs,          /* number of 32-bit limbs per value */
    __global uint *p_out,      /* [num_terms][nlimbs] flattened   */
    __global uint *q_out,
    __global uint *t_out)
{
    int tid = get_global_id(0);
    if (tid >= num_terms) return;

    long k = first_k + tid;

    /* Compute P = (6k-5)(6k-1)(2k-1) — fits in 64-bit for any
       reasonable k, so we store it in limb[0]. */
    long km6_5 = 6L * k - 5L;
    long km6_1 = 6L * k - 1L;
    long km2_1 = 2L * k - 1L;
    long long p_val = (long long)km6_5 * (long long)km6_1 * (long long)km2_1;

    /* Compute Q = (C^3/24) * k^3 — fits in 64-bit for k < ~2M */
    long long k3 = (long long)k * (long long)k * (long long)k;
    long long q_val = (long long)CHD_C3_OVER_24 * k3;

    /* Compute T = (A + B*k) * P * (-1)^k */
    long long ak_b = (long long)CHD_A + (long long)CHD_B * (long long)k;
    long long t_val = ak_b * p_val;
    if (k & 1) t_val = -t_val;

    /* Store into limb arrays (little-endian, 32-bit limbs) */
    int base = tid * nlimbs;

    /* P */
    ulong up = (ulong)(p_val >= 0 ? p_val : -p_val);
    p_out[base]     = (uint)(up & 0xFFFFFFFFUL);
    p_out[base + 1] = (uint)((up >> 32) & 0xFFFFFFFFUL);
    for (int i = 2; i < nlimbs; i++) p_out[base + i] = 0;

    /* Q */
    ulong uq = (ulong)(q_val >= 0 ? q_val : -q_val);
    q_out[base]     = (uint)(uq & 0xFFFFFFFFUL);
    q_out[base + 1] = (uint)((uq >> 32) & 0xFFFFFFFFUL);
    q_out[base + 2] = (uint)((uq >> 64) & 0xFFFFFFFFUL);
    for (int i = 3; i < nlimbs; i++) q_out[base + i] = 0;

    /* T — may need more limbs */
    ulong ut = (ulong)(t_val >= 0 ? t_val : -t_val);
    t_out[base]     = (uint)(ut & 0xFFFFFFFFUL);
    t_out[base + 1] = (uint)((ut >> 32) & 0xFFFFFFFFUL);
    t_out[base + 2] = (uint)((ut >> 64) & 0xFFFFFFFFUL);
    t_out[base + 3] = (uint)((ut >> 96) & 0xFFFFFFFFUL);
    for (int i = 4; i < nlimbs; i++) t_out[base + i] = 0;
}

)";

/* ─── OpenCL helper structures ──────────────────────────────── */

typedef struct {
    cl_platform_id   platform;
    cl_device_id     device;
    cl_context       context;
    cl_command_queue queue;
    cl_program       program;
    cl_kernel        k_eval_leaf;
    cl_kernel        k_big_mul;
    cl_kernel        k_propagate;
    cl_ulong         local_mem_size;
    cl_ulong         max_mem_alloc;
    size_t           max_wg_size;
    int              compute_units;
} gpu_ctx_t;

/* ─── OpenCL error checking macro ───────────────────────────── */

#define CL_CHECK(err, msg) do {                                         \
    if ((err) != CL_SUCCESS) {                                          \
        fprintf(stderr, "OpenCL error %d: %s (%s:%d)\n",               \
                (err), msg, __FILE__, __LINE__);                        \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while (0)

/* ─── GPU initialization ─────────────────────────────────────── */

static int gpu_init(gpu_ctx_t *g)
{
    cl_int err;

    /* Prefer GPU device */
    err = clGetPlatformIDs(1, &g->platform, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "No OpenCL platform found.\n");
        return -1;
    }

    err = clGetDeviceIDs(g->platform, CL_DEVICE_TYPE_GPU, 1,
                         &g->device, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "No GPU device found. Falling back to CPU.\n");
        err = clGetDeviceIDs(g->platform, CL_DEVICE_TYPE_CPU, 1,
                             &g->device, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "No OpenCL device at all.\n");
            return -1;
        }
    }

    /* Print device name */
    char dev_name[256];
    clGetDeviceInfo(g->device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
    printf(" OpenCL device: %s\n", dev_name);

    clGetDeviceInfo(g->device, CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof(int), &g->compute_units, NULL);
    clGetDeviceInfo(g->device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                    sizeof(cl_ulong), &g->max_mem_alloc, NULL);
    clGetDeviceInfo(g->device, CL_DEVICE_LOCAL_MEM_SIZE,
                    sizeof(cl_ulong), &g->local_mem_size, NULL);
    clGetDeviceInfo(g->device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                    sizeof(size_t), &g->max_wg_size, NULL);

    printf(" Compute units:  %d\n", g->compute_units);
    printf(" Max mem alloc:  %.1f MB\n",
           (double)g->max_mem_alloc / (1024.0 * 1024.0));

    g->context = clCreateContext(NULL, 1, &g->device, NULL, NULL, &err);
    CL_CHECK(err, "clCreateContext");

    g->queue = clCreateCommandQueueWithProperties(g->context, g->device,
                                                   0, &err);
    CL_CHECK(err, "clCreateCommandQueue");

    g->program = clCreateProgramWithSource(g->context, 1, &kernel_source,
                                            NULL, &err);
    CL_CHECK(err, "clCreateProgramWithSource");

    err = clBuildProgram(g->program, 1, &g->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(g->program, g->device, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &log_size);
        char *log = malloc(log_size);
        clGetProgramBuildInfo(g->program, g->device, CL_PROGRAM_BUILD_LOG,
                              log_size, log, NULL);
        fprintf(stderr, "Build log:\n%s\n", log);
        free(log);
        return -1;
    }

    g->k_eval_leaf = clCreateKernel(g->program, "eval_leaf_terms", &err);
    CL_CHECK(err, "clCreateKernel(eval_leaf)");
    g->k_big_mul = clCreateKernel(g->program, "big_mul", &err);
    CL_CHECK(err, "clCreateKernel(big_mul)");
    g->k_propagate = clCreateKernel(g->program, "propagate_carry", &err);
    CL_CHECK(err, "clCreateKernel(propagate_carry)");

    return 0;
}

static void gpu_cleanup(gpu_ctx_t *g)
{
    clReleaseKernel(g->k_eval_leaf);
    clReleaseKernel(g->k_big_mul);
    clReleaseKernel(g->k_propagate);
    clReleaseProgram(g->program);
    clReleaseCommandQueue(g->queue);
    clReleaseContext(g->context);
}

/* ─── RAM detection ──────────────────────────────────────────── */

static int64_t get_available_memory_mb(void)
{
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) {
        fprintf(stderr, "[WARN] Cannot read /proc/meminfo; assuming 8 GB.\n");
        return 8192;
    }

    char line[256];
    int64_t kb = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, "MemAvailable:")) {
            sscanf(line + 13, "%ld", &kb);
            break;
        }
    }
    fclose(f);

    if (kb == 0) {
        fprintf(stderr, "[WARN] MemAvailable not found; assuming 8 GB.\n");
        return 8192;
    }
    return kb / 1024;  /* convert KB to MB */
}

/* ─── Timing ─────────────────────────────────────────────────── */

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ─── GMP-based binary splitting (CPU fallback for small ranges) ── */

static void bs_cpu(int64_t a, int64_t b, mpz_t P, mpz_t Q, mpz_t T)
{
    if (b - a == 1) {
        int64_t k = a;
        int64_t km6_5 = 6*k - 5;
        int64_t km6_1 = 6*k - 1;
        int64_t km2_1 = 2*k - 1;

        mpz_set_si(P, km6_5);
        mpz_mul_si(P, P, km6_1);
        mpz_mul_si(P, P, km2_1);

        mpz_set_si(Q, k);
        mpz_mul_si(Q, Q, k);
        mpz_mul_si(Q, Q, k);
        mpz_mul_si(Q, Q, CHD_C3_OVER_24);

        mpz_set_si(T, CHD_A);
        mpz_addmul_si(T, Q, 0);  /* placeholder */
        /* T = (A + B*k) * P */
        mpz_set_si(T, CHD_B);
        mpz_mul_si(T, T, k);
        mpz_add_si(T, T, CHD_A);
        mpz_mul(T, T, P);

        if (k & 1) mpz_neg(T, T);
        return;
    }

    int64_t m = (a + b) / 2;
    mpz_t PL, QL, TL, PR, QR, TR;
    mpz_init(PL); mpz_init(QL); mpz_init(TL);
    mpz_init(PR); mpz_init(QR); mpz_init(TR);

    bs_cpu(a, m, PL, QL, TL);
    bs_cpu(m, b, PR, QR, TR);

    mpz_mul(P, PL, PR);
    mpz_mul(Q, QL, QR);

    mpz_t tmp1, tmp2;
    mpz_init(tmp1); mpz_init(tmp2);
    mpz_mul(tmp1, TL, QR);
    mpz_mul(tmp2, PL, TR);
    mpz_add(T, tmp1, tmp2);

    mpz_clear(tmp1); mpz_clear(tmp2);
    mpz_clear(PL); mpz_clear(QL); mpz_clear(TL);
    mpz_clear(PR); mpz_clear(QR); mpz_clear(TR);
}

/* ─── Main ───────────────────────────────────────────────────── */

int main(void)
{
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║   GPU-Accelerated Pi Calculator                  ║\n");
    printf("║   Chudnovsky + Binary Splitting + OpenCL         ║\n");
    printf("║   Optimized for Steam Deck APU (AMD RDNA 2)      ║\n");
    printf("╚══════════════════════════════════════════════════╝\n\n");

    /* ── 1. Detect memory ── */
    int64_t avail_mb = get_available_memory_mb();
    double avail_bytes = (double)avail_mb * 1024.0 * 1024.0;

    /* ── 2. Estimate digits ── */
    double mem_budget = MEM_BUDGET_FRAC * avail_bytes;
    int64_t num_digits = 10000;
    int64_t nterms;
    double log2n, est_per_digit;

    for (int iter = 0; iter < 30; iter++) {
        nterms = (int64_t)ceil((double)num_digits / DIGITS_PER_TERM) + 10;
        log2n = log2((double)nterms);
        est_per_digit = log2n * 1.64;
        num_digits = (int64_t)(mem_budget / est_per_digit);
        if (num_digits < 1000) { num_digits = 1000; break; }
        if (num_digits > MAX_DIGITS_CAP) { num_digits = MAX_DIGITS_CAP; break; }
    }
    num_digits -= num_digits % 100;
    nterms = (int64_t)ceil((double)num_digits / DIGITS_PER_TERM) + 10;

    int nthreads = 1;
    #pragma omp parallel
    #pragma omp single
    nthreads = omp_get_num_threads();

    printf(" Available RAM:      %ld MB\n", (long)avail_mb);
    printf(" Target digits:      %ld\n", (long)num_digits);
    printf(" Chudnovsky terms:   %ld\n", (long)nterms);
    printf(" OpenMP threads:     %d\n", nthreads);

    /* ── 3. Initialize GPU ── */
    gpu_ctx_t gpu = {0};
    int gpu_ok = (gpu_init(&gpu) == 0);
    if (!gpu_ok) {
        printf(" GPU not available; falling back to CPU-only mode.\n");
    }
    printf("\n");

    /* ── 4. Compute ── */
    printf("Computing Pi ...\n");
    double t_start = now_sec();

    mpz_t P, Q, Tval;
    mpz_init(P); mpz_init(Q); mpz_init(Tval);

    if (gpu_ok && nterms > SERIAL_CUTOFF) {
        /*
         * Hybrid approach:
         *   - Split the term range into large chunks
         *   - For chunks above SERIAL_CUTOFF, use OpenMP tasks
         *     with GPU offload for leaf evaluation
         *   - For small chunks, use CPU binary splitting
         *
         * For simplicity and correctness, we use the CPU binary
         * splitting for the tree structure, but evaluate leaf
         * batches on the GPU when the range is large enough.
         */

        /* Number of leaf batches */
        int64_t nbatches = (nterms + GPU_BATCH_SIZE - 1) / GPU_BATCH_SIZE;
        printf(" GPU leaf batches:   %ld (batch size %d)\n",
               (long)nbatches, GPU_BATCH_SIZE);

        /* Allocate result arrays for all batches */
        /* Each leaf produces P, Q, T as GMP integers.
           We'll compute leaves on GPU, then merge on CPU. */

        /* For the Steam Deck's unified memory, we can use
           CL_MEM_ALLOC_HOST_PTR for zero-copy access. */

        /* --- Phase 1: Evaluate all leaf terms on GPU --- */
        double t_gpu_start = now_sec();

        /* Limb count for leaf values: Q can be up to
           C^3/24 * k^3 where k ~ nterms.
           log2(Q_max) ≈ log2(1e16) + 3*log2(nterms)
           For nterms ~ 1M: ~53 + 60 = 113 bits → 4 x 32-bit limbs */
        int leaf_nlimbs = 8;  /* generous for leaf values */

        size_t batch_bytes = (size_t)GPU_BATCH_SIZE * leaf_nlimbs * sizeof(uint);

        cl_int err;
        cl_mem buf_p = clCreateBuffer(gpu.context,
                            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                            batch_bytes, NULL, &err);
        cl_mem buf_q = clCreateBuffer(gpu.context,
                            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                            batch_bytes, NULL, &err);
        cl_mem buf_t = clCreateBuffer(gpu.context,
                            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                            batch_bytes, NULL, &err);

        /* Allocate GMP arrays for leaf results */
        mpz_t *leaf_p = malloc((size_t)nbatches * GPU_BATCH_SIZE * sizeof(mpz_t));
        mpz_t *leaf_q = malloc((size_t)nbatches * GPU_BATCH_SIZE * sizeof(mpz_t));
        mpz_t *leaf_t = malloc((size_t)nbatches * GPU_BATCH_SIZE * sizeof(mpz_t));

        for (int64_t i = 0; i < nbatches * GPU_BATCH_SIZE; i++) {
            mpz_init(leaf_p[i]);
            mpz_init(leaf_q[i]);
            mpz_init(leaf_t[i]);
        }

        printf(" Evaluating leaf terms on GPU ...\n");

        for (int64_t batch = 0; batch < nbatches; batch++) {
            int64_t k_start = batch * GPU_BATCH_SIZE;
            int terms_this_batch = GPU_BATCH_SIZE;
            if (k_start + terms_this_batch > nterms)
                terms_this_batch = (int)(nterms - k_start);

            clSetKernelArg(gpu.k_eval_leaf, 0, sizeof(long), &k_start);
            clSetKernelArg(gpu.k_eval_leaf, 1, sizeof(int), &terms_this_batch);
            clSetKernelArg(gpu.k_eval_leaf, 2, sizeof(int), &leaf_nlimbs);
            clSetKernelArg(gpu.k_eval_leaf, 3, sizeof(cl_mem), &buf_p);
            clSetKernelArg(gpu.k_eval_leaf, 4, sizeof(cl_mem), &buf_q);
            clSetKernelArg(gpu.k_eval_leaf, 5, sizeof(cl_mem), &buf_t);

            size_t global_size = (size_t)terms_this_batch;
            err = clEnqueueNDRangeKernel(gpu.queue, gpu.k_eval_leaf,
                                          1, NULL, &global_size, NULL,
                                          0, NULL, NULL);
            CL_CHECK(err, "enqueue eval_leaf");

            /* Read back results */
            uint *hp = (uint*)clEnqueueMapBuffer(gpu.queue, buf_p, CL_TRUE,
                            CL_MAP_READ, 0, batch_bytes, 0, NULL, NULL, &err);
            uint *hq = (uint*)clEnqueueMapBuffer(gpu.queue, buf_q, CL_TRUE,
                            CL_MAP_READ, 0, batch_bytes, 0, NULL, NULL, &err);
            uint *ht = (uint*)clEnqueueMapBuffer(gpu.queue, buf_t, CL_TRUE,
                            CL_MAP_READ, 0, batch_bytes, 0, NULL, NULL, &err);

            /* Convert limb arrays to GMP integers */
            for (int i = 0; i < terms_this_batch; i++) {
                int base = i * leaf_nlimbs;
                mpz_set_ui(leaf_p[batch * GPU_BATCH_SIZE + i], 0);
                mpz_set_ui(leaf_q[batch * GPU_BATCH_SIZE + i], 0);
                mpz_set_ui(leaf_t[batch * GPU_BATCH_SIZE + i], 0);

                for (int j = leaf_nlimbs - 1; j >= 0; j--) {
                    mpz_mul_2exp(leaf_p[batch*GPU_BATCH_SIZE+i],
                                 leaf_p[batch*GPU_BATCH_SIZE+i], 32);
                    mpz_add_ui(leaf_p[batch*GPU_BATCH_SIZE+i],
                               leaf_p[batch*GPU_BATCH_SIZE+i],
                               (ulong)hp[base + j]);
                    mpz_mul_2exp(leaf_q[batch*GPU_BATCH_SIZE+i],
                                 leaf_q[batch*GPU_BATCH_SIZE+i], 32);
                    mpz_add_ui(leaf_q[batch*GPU_BATCH_SIZE+i],
                               leaf_q[batch*GPU_BATCH_SIZE+i],
                               (ulong)hq[base + j]);
                    mpz_mul_2exp(leaf_t[batch*GPU_BATCH_SIZE+i],
                                 leaf_t[batch*GPU_BATCH_SIZE+i], 32);
                    mpz_add_ui(leaf_t[batch*GPU_BATCH_SIZE+i],
                               leaf_t[batch*GPU_BATCH_SIZE+i],
                               (ulong)ht[base + j]);
                }
            }

            clEnqueueUnmapMemObject(gpu.queue, buf_p, hp, 0, NULL, NULL);
            clEnqueueUnmapMemObject(gpu.queue, buf_q, hq, 0, NULL, NULL);
            clEnqueueUnmapMemObject(gpu.queue, buf_t, ht, 0, NULL, NULL);
        }

        clFinish(gpu.queue);
        double t_gpu_end = now_sec();
        printf(" GPU leaf eval:      %.3f s\n", t_gpu_end - t_gpu_start);

        clReleaseMemObject(buf_p);
        clReleaseMemObject(buf_q);
        clReleaseMemObject(buf_t);

        /* --- Phase 2: Merge leaf results on CPU using binary splitting --- */
        printf(" Merging results on CPU ...\n");

        /* We now have nterms leaf values in arrays.
           Merge them using a parallel reduction tree. */

        int64_t n = nterms;
        mpz_t *mp = leaf_p, *mq = leaf_q, *mt = leaf_t;

        for (int64_t stride = 1; stride < n; stride *= 2) {
            int64_t pairs = n / (2 * stride);
            if (pairs == 0) break;

            #pragma omp parallel for schedule(dynamic, 64)
            for (int64_t i = 0; i < pairs; i++) {
                int64_t li = 2 * i * stride;
                int64_t ri = (2 * i + 1) * stride;

                mpz_t new_p, new_q, new_t, tmp1, tmp2;
                mpz_init(new_p); mpz_init(new_q); mpz_init(new_t);
                mpz_init(tmp1); mpz_init(tmp2);

                mpz_mul(new_p, mp[li], mp[ri]);
                mpz_mul(new_q, mq[li], mq[ri]);
                mpz_mul(tmp1, mt[li], mq[ri]);
                mpz_mul(tmp2, mp[li], mt[ri]);
                mpz_add(new_t, tmp1, tmp2);

                mpz_set(mp[li], new_p);
                mpz_set(mq[li], new_q);
                mpz_set(mt[li], new_t);

                mpz_clear(new_p); mpz_clear(new_q); mpz_clear(new_t);
                mpz_clear(tmp1); mpz_clear(tmp2);
            }
        }

        mpz_set(P, mp[0]);
        mpz_set(Q, mq[0]);
        mpz_set(Tval, mt[0]);

        /* Cleanup leaf arrays */
        for (int64_t i = 0; i < nbatches * GPU_BATCH_SIZE; i++) {
            mpz_clear(leaf_p[i]);
            mpz_clear(leaf_q[i]);
            mpz_clear(leaf_t[i]);
        }
        free(leaf_p);
        free(leaf_q);
        free(leaf_t);

    } else {
        /* CPU-only path */
        printf(" Using CPU-only binary splitting ...\n");
        bs_cpu(0, nterms, P, Q, Tval);
    }

    double t_end = now_sec();
    double t_elapsed = t_end - t_start;

    printf(" Binary splitting:   %.3f s\n", t_elapsed);

    /* ── 5. Final assembly: Pi = D * sqrt(E) * Q / (A*Q + T) ── */
    printf("Final assembly ...\n");

    int64_t extra_digits = 50;
    mpz_t pow10, sqrt_val, num, den, pi_val;
    mpz_init(pow10);
    mpz_init(sqrt_val);
    mpz_init(num);
    mpz_init(den);
    mpz_init(pi_val);

    mpz_ui_pow_ui(pow10, 10, (ulong)(num_digits + extra_digits));
    mpz_mul_ui(sqrt_val, pow10, CHD_E);
    mpz_mul(sqrt_val, sqrt_val, pow10);
    mpz_sqrt(sqrt_val, sqrt_val);

    mpz_mul(num, Q, sqrt_val);
    mpz_mul_ui(num, num, CHD_D);

    mpz_mul_ui(den, Q, CHD_A);
    mpz_add(den, den, Tval);

    mpz_fdiv_q(pi_val, num, den);

    /* Count digits */
    char *pi_str = mpz_get_str(NULL, 10, pi_val);
    int64_t actual_digits = (int64_t)strlen(pi_str) - 1;  /* minus leading '3' */

    /* ── 6. Results ── */
    printf("\n");
    printf("╔══════════════════════════════════════════════════╗\n");
    printf("║   RESULTS                                        ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");
    printf(" Digits of Pi:       %ld\n", (long)actual_digits);
    printf(" Total time:         %.3f s\n", t_elapsed);
    printf(" Threads used:       %d\n", nthreads);
    printf(" Memory available:   %ld MB\n", (long)avail_mb);
    if (gpu_ok) printf(" GPU acceleration:   Yes (OpenCL)\n");
    else         printf(" GPU acceleration:   No (CPU only)\n");
    printf("\n");

    printf(" Pi (first 80 decimal places):\n");
    printf("  3.");
    for (int i = 1; i <= 80 && pi_str[i]; i++) {
        putchar(pi_str[i]);
        if (i % 10 == 0) putchar(' ');
        if (i % 40 == 0) printf("\n   ");
    }
    printf("...\n\n");

    /* Write to file */
    char fname[128];
    snprintf(fname, sizeof(fname), "pi_%ld_digits.txt", (long)actual_digits);
    FILE *fout = fopen(fname, "w");
    if (fout) {
        fprintf(fout, "%s\n", pi_str);
        fclose(fout);
        printf(" Full result written to: %s\n", fname);
    }

    free(pi_str);
    mpz_clear(pow10); mpz_clear(sqrt_val);
    mpz_clear(num); mpz_clear(den);
    mpz_clear(pi_val);
    mpz_clear(P); mpz_clear(Q); mpz_clear(Tval);

    if (gpu_ok) gpu_cleanup(&gpu);

    printf("\nDone.\n");
    return 0;
}
