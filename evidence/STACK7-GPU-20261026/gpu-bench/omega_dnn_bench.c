/* SPDX-License-Identifier: Apache-2.0
 * omega_dnn_bench: THROWAWAY measurement driver (lane OmegaDnn.GpuDnn, 2026-10-26), not product code.
 * Links omega-mv e5b32567a build/libomega_gpu.a (INTERIM: C codegen emitting SASS, not .om).
 * Usage: omega_dnn_bench OUTDIR [--mutant] [--warm 10] [--reps 50]
 * For every predeclared case: generate inputs with a fixed LCG, write inputs + first output to OUTDIR,
 * run `warm` untimed calls then `reps` timed calls; per timed call record the API wall time (clock_gettime
 * around the synchronous call: staging, launch, marker wait, copy-out) and info.elapsed_ns (submit to
 * completion marker). Every timed output is compared bitwise with the first output. One JSON line per case.
 * --mutant: arm the library's TEST-ONLY mutant kernel for each op (negative control), one call, no timing. */
#include "omega_gpu_elementwise_api.h"
#include "omega_gpu_attention_api.h"
#include "omega_gpu_session.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }
static float frand(uint32_t *s, float lo, float hi) { return lo + (hi - lo) * ((float)(lcg(s) >> 8) / 16777216.0f); }
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); uint32_t r = ((u >> 16) & 1u) + 0x7fffu; return (uint16_t)((u + r) >> 16); }
static uint64_t now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }

static const char *g_dir;
static void dump(const char *cs, const char *what, const void *p, size_t bytes) {
    char path[512]; snprintf(path, sizeof path, "%s/%s.%s.bin", g_dir, cs, what);
    FILE *f = fopen(path, "wb"); if (!f || fwrite(p, 1, bytes, f) != bytes) { fprintf(stderr, "write %s failed\n", path); exit(3); } fclose(f);
}

typedef struct {
    int kind; /* 0 rmsnorm 1 rope 2 swiglu 3 gqa 4 rows 5 paged */
    uint32_t a, b, c, d, e; /* shape words, meaning per kind */
    float *x, *w, *y; void *pool; uint32_t *ids; size_t out_n;
} Case;

static int call(Case *k, float *out, uint64_t *chip_ns) {
    int rc;
    if (k->kind <= 2) {
        OmegaGpuEwInfo info; memset(&info, 0, sizeof info);
        if (k->kind == 0) rc = omega_gpu_rmsnorm_f32(k->a, k->b, k->x, k->w, 1e-5f, out, &info);
        else if (k->kind == 1) rc = omega_gpu_rope_f32(k->a, k->b, k->x, k->w, k->y, out, &info);
        else rc = omega_gpu_swiglu_f32(k->a, k->x, k->w, out, &info);
        *chip_ns = info.elapsed_ns; return rc;
    }
    OmegaGpuAttnInfo info; memset(&info, 0, sizeof info);
    if (k->kind == 3) rc = omega_gpu_gqa_attention_f32(k->x, k->w, k->y, k->d, k->a, k->b, k->c, out, &info);
    else if (k->kind == 4) rc = omega_gpu_gqa_attention_rows_f32(k->x, k->w, k->y, 0, k->d, k->a, k->b, k->c, out, &info);
    else {
        /* resident pool: block_size e, one layer */
        OmegaGpuKvLayout ly; uint32_t hd = k->c, nkv = k->b, bs = k->e, nblk = (k->d + bs - 1) / bs;
        ly.head_stride_bytes = (uint64_t)hd * 2; ly.token_stride_bytes = ly.head_stride_bytes * nkv;
        ly.kv_plane_stride_bytes = ly.token_stride_bytes * bs; ly.layer_stride_bytes = 2 * ly.kv_plane_stride_bytes;
        ly.block_stride_bytes = ly.layer_stride_bytes; ly.pool_bytes = ly.block_stride_bytes * nblk;
        ly.num_blocks = nblk; ly.num_layers = 1; ly.block_size = bs;
        rc = omega_gpu_paged_attention_bf16(k->x, k->pool, &ly, k->ids, nblk, k->d, 0, k->a, k->b, hd, out, &info);
        if (rc == 0 && info.kv_source != OMEGA_GPU_ATTN_KV_RESIDENT) { fprintf(stderr, "paged call did not take the resident path\n"); return -99; }
    }
    *chip_ns = info.elapsed_ns; return rc;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s OUTDIR [--mutant] [--warm N] [--reps N]\n", argv[0]); return 2; }
    g_dir = argv[1]; int mutant = 0, warm = 10, reps = 50;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--mutant")) mutant = 1;
        else if (!strcmp(argv[i], "--warm") && i + 1 < argc) warm = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        /* POST-HOC reference-only arm (not in the frozen protocol): marker-poll spin window, omega#328 */
        else if (!strcmp(argv[i], "--spin-us") && i + 1 < argc) {
            if (omega_gpu_session_set_spin_us((uint32_t)atoi(argv[++i])) != 0) { fprintf(stderr, "spin_us refused\n"); return 2; }
        }
    }
    if (mutant) { warm = 0; reps = 0; }
    /* predeclared cases (GPU.md section 2); names are parsed by the Python side */
    static const struct { const char *name; int kind; uint32_t a, b, c, d, e; } C[] = {
        { "rmsnorm_1x2048", 0, 1, 2048, 0, 0, 0 },
        { "rmsnorm_64x2048", 0, 64, 2048, 0, 0, 0 },
        { "rope_32x128", 1, 32, 128, 0, 0, 0 },
        { "swiglu_5632", 2, 5632, 0, 0, 0, 0 },
        { "swiglu_131072", 2, 131072, 0, 0, 0, 0 },
        { "gqa_f32_32q8kv_hd128_ctx2048", 3, 32, 8, 128, 2048, 0 },
        { "gqa_f32_32q4kv_hd64_ctx2048", 3, 32, 4, 64, 2048, 0 },
        { "gqa_rows_f32_32q8kv_hd128_m256", 4, 32, 8, 128, 256, 0 },
        { "paged_bf16_32q8kv_hd128_ctx2048_bs16", 5, 32, 8, 128, 2048, 16 },
    };
    uint32_t seed = 0x0d1ec0deu;
    for (size_t ci = 0; ci < sizeof C / sizeof C[0]; ci++) {
        Case k; memset(&k, 0, sizeof k);
        k.kind = C[ci].kind; k.a = C[ci].a; k.b = C[ci].b; k.c = C[ci].c; k.d = C[ci].d; k.e = C[ci].e;
        const char *cs = C[ci].name;
        if (k.kind == 0) {
            size_t n = (size_t)k.a * k.b; k.out_n = n;
            k.x = malloc(n * 4); k.w = malloc(k.b * 4);
            for (size_t i = 0; i < n; i++) k.x[i] = frand(&seed, -3.0f, 3.0f);
            for (size_t i = 0; i < k.b; i++) k.w[i] = frand(&seed, 0.5f, 1.5f);
            dump(cs, "x", k.x, n * 4); dump(cs, "w", k.w, k.b * 4);
        } else if (k.kind == 1) {
            size_t n = (size_t)k.a * k.b, half = k.b / 2; k.out_n = n;
            k.x = malloc(n * 4); k.w = malloc(half * 4); k.y = malloc(half * 4);
            for (size_t i = 0; i < n; i++) k.x[i] = frand(&seed, -1.0f, 1.0f);
            for (size_t i = 0; i < half; i++) { double rot = 1234.0 / pow(1e6, (double)(2 * i) / (double)k.b); k.w[i] = (float)cos(rot); k.y[i] = (float)sin(rot); }
            dump(cs, "x", k.x, n * 4); dump(cs, "cos", k.w, half * 4); dump(cs, "sin", k.y, half * 4);
        } else if (k.kind == 2) {
            size_t n = k.a; k.out_n = n;
            k.x = malloc(n * 4); k.w = malloc(n * 4);
            for (size_t i = 0; i < n; i++) { k.x[i] = frand(&seed, -3.0f, 3.0f); k.w[i] = frand(&seed, -2.0f, 2.0f); }
            dump(cs, "gate", k.x, n * 4); dump(cs, "up", k.w, n * 4);
        } else if (k.kind == 3 || k.kind == 4) {
            size_t rows = k.kind == 3 ? 1 : k.d, ctx = k.d;
            size_t nq = rows * k.a * k.c, nkv = ctx * k.b * k.c; k.out_n = nq;
            k.x = malloc(nq * 4); k.w = malloc(nkv * 4); k.y = malloc(nkv * 4);
            for (size_t i = 0; i < nq; i++) k.x[i] = frand(&seed, -1.0f, 1.0f);
            for (size_t i = 0; i < nkv; i++) { k.w[i] = frand(&seed, -1.0f, 1.0f); k.y[i] = frand(&seed, -1.0f, 1.0f); }
            dump(cs, "q", k.x, nq * 4); dump(cs, "k", k.w, nkv * 4); dump(cs, "v", k.y, nkv * 4);
        } else {
            uint32_t bs = k.e, nblk = (k.d + bs - 1) / bs; size_t nq = (size_t)k.a * k.c; k.out_n = nq;
            size_t pool_bytes = (size_t)nblk * 2 * bs * k.b * k.c * 2;
            if (omega_gpu_attention_pool_alloc(pool_bytes, &k.pool) != 0) { fprintf(stderr, "pool_alloc failed: %s\n", omega_gpu_attention_last_error()); return 4; }
            uint16_t *pw = k.pool;
            for (size_t i = 0; i < pool_bytes / 2; i++) pw[i] = f2bf(frand(&seed, -1.0f, 1.0f));
            k.ids = malloc(nblk * 4);
            for (uint32_t b = 0; b < nblk; b++) k.ids[b] = (b * 37u + 5u) % nblk; /* scattered, a permutation (37 odd, nblk power of two) */
            k.x = malloc(nq * 4);
            for (size_t i = 0; i < nq; i++) k.x[i] = frand(&seed, -1.0f, 1.0f);
            dump(cs, "q", k.x, nq * 4); dump(cs, "pool", k.pool, pool_bytes); dump(cs, "ids", k.ids, nblk * 4);
        }
        if (mutant) {
            if (k.kind == 0) omega_gpu_elementwise_test_set_mutant(OMEGA_GPU_EW_RMSNORM);
            else if (k.kind == 1) omega_gpu_elementwise_test_set_mutant(OMEGA_GPU_EW_ROPE);
            else if (k.kind == 2) omega_gpu_elementwise_test_set_mutant(OMEGA_GPU_EW_SWIGLU);
            else omega_gpu_attention_test_set_mutant(OMEGA_GPU_ATTN_MUTANT_KV_HEAD);
        }
        float *first = malloc(k.out_n * 4), *got = malloc(k.out_n * 4);
        uint64_t chip; uint64_t t0 = now_ns();
        int rc = call(&k, first, &chip);
        uint64_t first_ns = now_ns() - t0;
        if (mutant) { omega_gpu_elementwise_test_set_mutant(0); omega_gpu_attention_test_set_mutant(OMEGA_GPU_ATTN_MUTANT_NONE); }
        dump(cs, "out", first, k.out_n * 4);
        int bad_rc = rc != 0, diff = 0;
        for (int i = 0; i < warm && !bad_rc; i++) { if (call(&k, got, &chip) != 0) bad_rc = 1; }
        uint64_t *wall = calloc((size_t)reps + 1, 8), *chp = calloc((size_t)reps + 1, 8);
        for (int i = 0; i < reps && !bad_rc; i++) {
            uint64_t s = now_ns(); int r = call(&k, got, &chp[i]); wall[i] = now_ns() - s;
            if (r != 0) bad_rc = 1;
            if (memcmp(got, first, k.out_n * 4) != 0) diff++;
        }
        printf("{\"case\":\"%s\",\"rc\":%d,\"mutant\":%d,\"first_call_ns\":%llu,\"warm\":%d,\"reps\":%d,\"timed_outputs_differing_from_first\":%d,\"wall_ns\":[",
               cs, rc, mutant, (unsigned long long)first_ns, warm, reps, diff);
        for (int i = 0; i < reps; i++) printf("%s%llu", i ? "," : "", (unsigned long long)wall[i]);
        printf("],\"chip_ns\":[");
        for (int i = 0; i < reps; i++) printf("%s%llu", i ? "," : "", (unsigned long long)chp[i]);
        printf("]}\n"); fflush(stdout);
        if (rc != 0) fprintf(stderr, "%s rc=%d err=%s\n", cs, rc, k.kind <= 2 ? omega_gpu_elementwise_last_error() : omega_gpu_attention_last_error());
        free(first); free(got); free(wall); free(chp); free(k.x); free(k.w); free(k.y); free(k.ids);
        if (k.pool) omega_gpu_attention_pool_free();
    }
    return 0;
}
