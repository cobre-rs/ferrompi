// Verbatim copy of ferrompi.c request-table alloc/free (lines 91-93, 296-349), MPI_Request=int.
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
typedef int MPI_Request;
#define MPI_REQUEST_NULL 0x2c000000
#define MAX_REQUESTS 16384
#define REQUEST_BITS_WORDS (MAX_REQUESTS / 64)
static MPI_Request request_table[MAX_REQUESTS];
static atomic_int next_request_hint;
static _Atomic(uint64_t) request_bits[REQUEST_BITS_WORDS];
#ifdef PLAIN
// Hypothetical non-atomic variant (valid only below MPI_THREAD_MULTIPLE)
static int64_t alloc_request(MPI_Request req) {
    unsigned hint = (unsigned)atomic_load_explicit(&next_request_hint, memory_order_relaxed);
    for (int w = 0; w < REQUEST_BITS_WORDS; w++) {
        unsigned widx = (hint + (unsigned)w) % REQUEST_BITS_WORDS;
        uint64_t cur = atomic_load_explicit(&request_bits[widx], memory_order_relaxed);
        if (cur != UINT64_MAX) {
            int bit = __builtin_ctzll(~cur);
            atomic_store_explicit(&request_bits[widx], cur | ((uint64_t)1 << bit), memory_order_relaxed);
            int64_t idx = (int64_t)widx * 64 + bit;
            request_table[idx] = req;
            if ((unsigned)hint != widx) atomic_store_explicit(&next_request_hint, (int)widx, memory_order_relaxed);
            return idx;
        }
    }
    return -1;
}
static void free_request(int64_t handle) {
    request_table[handle] = MPI_REQUEST_NULL;
    unsigned widx = (unsigned)(handle / 64);
    uint64_t cur = atomic_load_explicit(&request_bits[widx], memory_order_relaxed);
    atomic_store_explicit(&request_bits[widx], cur & ~((uint64_t)1 << (handle % 64)), memory_order_release);
}

#elif defined(DENSE)
// Pre-bitmap design per ADR-0002 Option A (dense atomic_int used[] + CAS scan from hint)
static atomic_int request_used[MAX_REQUESTS];
static int64_t alloc_request(MPI_Request req) {
    int hint = atomic_load_explicit(&next_request_hint, memory_order_relaxed);
    for (int i = 0; i < MAX_REQUESTS; i++) {
        int idx = (hint + i) % MAX_REQUESTS;
        int expected = 0;
        if (atomic_compare_exchange_strong_explicit(&request_used[idx], &expected, 1,
                memory_order_acq_rel, memory_order_relaxed)) {
            request_table[idx] = req;
            atomic_store_explicit(&next_request_hint, (idx + 1) % MAX_REQUESTS, memory_order_relaxed);
            return idx;
        }
    }
    return -1;
}
static void free_request(int64_t handle) {
    request_table[handle] = MPI_REQUEST_NULL;
    atomic_store_explicit(&request_used[handle], 0, memory_order_release);
}
#else
static int64_t alloc_request(MPI_Request req) {
    unsigned hint = (unsigned)atomic_load_explicit(&next_request_hint, memory_order_relaxed);
    for (int w = 0; w < REQUEST_BITS_WORDS; w++) {
        unsigned widx = (hint + (unsigned)w) % REQUEST_BITS_WORDS;
        uint64_t cur = atomic_load_explicit(&request_bits[widx], memory_order_relaxed);
        while (cur != UINT64_MAX) {
            int bit = __builtin_ctzll(~cur);
            uint64_t mask = (uint64_t)1 << bit;
            uint64_t old = atomic_fetch_or_explicit(&request_bits[widx], mask, memory_order_acq_rel);
            if ((old & mask) == 0) {
                int64_t idx = (int64_t)widx * 64 + bit;
                request_table[idx] = req;
                atomic_store_explicit(&next_request_hint, (int)widx, memory_order_relaxed);
                return idx;
            }
            cur = old;
        }
    }
    return -1;
}
static void free_request(int64_t handle) {
    if (handle >= 0 && handle < MAX_REQUESTS) {
        request_table[handle] = MPI_REQUEST_NULL;
        unsigned widx = (unsigned)(handle / 64);
        uint64_t mask = (uint64_t)1 << (handle % 64);
        atomic_fetch_and_explicit(&request_bits[widx], ~mask, memory_order_release);
    }
}
#endif
static long ITERS = 2000000; static int INFLIGHT = 16;
static pthread_barrier_t bar;
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e9+t.tv_nsec; }
static void* worker(void* arg) {
    int64_t h[64]; volatile int64_t sink = 0;
    pthread_barrier_wait(&bar);
    double t0 = now();
    for (long it = 0; it < ITERS / INFLIGHT; it++) {
        for (int i = 0; i < INFLIGHT; i++) h[i] = alloc_request(i);
        for (int i = 0; i < INFLIGHT; i++) { sink += h[i]; free_request(h[i]); }
    }
    *(double*)arg = (now() - t0) / (double)ITERS;
    return NULL;
}
int main(int argc, char** argv) {
    int T = argc > 1 ? atoi(argv[1]) : 1;
    pthread_t th[64]; double r[64];
    pthread_barrier_init(&bar, NULL, T);
    for (int i = 0; i < T; i++) pthread_create(&th[i], NULL, worker, &r[i]);
    double mx = 0, sum = 0;
    for (int i = 0; i < T; i++) { pthread_join(th[i], NULL); sum += r[i]; if (r[i] > mx) mx = r[i]; }
    printf("T=%-2d alloc+free pair: mean per-thread %.1f ns (worst thread %.1f ns)\n", T, sum / T, mx);
    return 0;
}
