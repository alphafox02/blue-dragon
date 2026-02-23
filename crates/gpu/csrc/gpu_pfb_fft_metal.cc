/*
 * Copyright 2025-2026 CEMAXECUTER LLC
 *
 * GPU PFB + FFT for blue-dragon -- Metal backend (macOS)
 *
 * Metal polyphase channelizer + VkFFT inverse FFT (backend 5).
 * Double-buffered: GPU processes batch N while host fills batch N+1.
 *
 * Provides the same public C API as gpu_pfb_fft.c (OpenCL backend).
 */

#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#define VKFFT_BACKEND 5
#include "vkFFT.h"

#include "Foundation/Foundation.hpp"
#include "QuartzCore/QuartzCore.hpp"
#include "Metal/Metal.hpp"

#define NUM_FFT 2

/* ------------------------------------------------------------------ */
/* Metal PFB kernel source (MSL, compiled at runtime)                  */
/* ------------------------------------------------------------------ */
static const char *pfb_kernel_source = R"(
#include <metal_stdlib>
using namespace metal;

kernel void pfb_channelize(
    device const char *raw_input      [[buffer(0)]],
    device const float *h_sub         [[buffer(1)]],
    device float *fft_buffer          [[buffer(2)]],
    constant uint &M                  [[buffer(3)]],
    constant uint &M2                 [[buffer(4)]],
    constant uint &h_sub_len          [[buffer(5)]],
    constant uint &pre_roll_steps     [[buffer(6)]],
    constant uint &batch_size         [[buffer(7)]],
    constant uint &initial_flag       [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint t = gid.x;
    uint ch = gid.y;
    if (t >= batch_size || ch >= M) return;

    uint abs_t = t + pre_roll_steps;
    uint flag_at_t = (initial_flag + abs_t) & 1u;

    uint sub_offset = flag_at_t
        ? ((M2 + ch) >= M ? (M2 + ch - M) : (M2 + ch))
        : ch;

    uint is_lower = (ch < M2) ? 1u : 0u;
    uint push_parity = is_lower
        ? (initial_flag & 1u)
        : ((initial_flag + 1u) & 1u);

    uint latest_push = ((abs_t & 1u) == push_parity)
        ? abs_t : (abs_t - 1u);

    uint ch_pos = is_lower
        ? (M2 - 1u - ch)
        : (M - 1u - ch);

    float real_sum = 0.0f, imag_sum = 0.0f;
    for (uint k = 0; k < h_sub_len; ++k) {
        uint push_step = latest_push - 2u * (h_sub_len - 1u - k);
        uint raw_idx = push_step * M2 + ch_pos;
        float sr = float(raw_input[raw_idx * 2u]);
        float si = float(raw_input[raw_idx * 2u + 1u]);
        float coeff = h_sub[sub_offset * h_sub_len + k];
        real_sum += sr * coeff;
        imag_sum += si * coeff;
    }

    float scale = 1.0f / 256.0f;
    uint out_idx = t * M + ch;
    fft_buffer[out_idx * 2u] = real_sum * scale;
    fft_buffer[out_idx * 2u + 1u] = imag_sum * scale;
}
)";

/* ------------------------------------------------------------------ */
/* Per-buffer context                                                  */
/* ------------------------------------------------------------------ */
typedef enum {
    BUFFER_STATE_READY,
    BUFFER_STATE_FILLING,
    BUFFER_STATE_EXECUTING,
    BUFFER_STATE_DONE,
} buffer_state_t;

typedef struct {
    VkFFTApplication app;

    MTL::CommandQueue *queue;

    /* GPU buffers (shared memory for CPU<->GPU) */
    MTL::Buffer *mtl_fft_buffer;
    MTL::Buffer *mtl_raw_buffer;

    uint64_t fft_buffer_size;
    uint64_t raw_buffer_size;

    buffer_state_t buffer_state;

    pthread_mutex_t mutex;
    pthread_cond_t state_cond;
} fft_context_t;

static fft_context_t fft_ctx[NUM_FFT];
static unsigned cur_fft = 0;

/* Metal globals */
static MTL::Device *mtl_device = nullptr;
static MTL::ComputePipelineState *pfb_pipeline = nullptr;
static MTL::Buffer *mtl_h_sub = nullptr;

/* PFB parameter buffers (constant args passed to shader) */
static MTL::Buffer *param_M = nullptr;
static MTL::Buffer *param_M2 = nullptr;
static MTL::Buffer *param_h_sub_len = nullptr;
static MTL::Buffer *param_pre_roll = nullptr;
static MTL::Buffer *param_batch_size = nullptr;
static MTL::Buffer *param_init_flag = nullptr;

/* PFB parameters */
static unsigned pfb_M, pfb_M2, pfb_h_sub_len;
static unsigned g_batch_size;
static unsigned pre_roll_steps;
static unsigned pre_roll_bytes;
static int8_t *pre_roll_buf = nullptr;

/* Worker thread */
static pthread_t worker_thread;
static volatile int gpu_running = 0;

/* Result delivery */
static float *result_ptr = nullptr;
static fft_context_t *result_ctx = nullptr;
static pthread_mutex_t result_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t result_cond = PTHREAD_COND_INITIALIZER;

static int first_submit = 1;

/* ------------------------------------------------------------------ */
/* GPU worker thread                                                   */
/* ------------------------------------------------------------------ */
static void *fft_worker(void *arg) {
    (void)arg;
    unsigned idx = 0;

    while (gpu_running) {
        fft_context_t *f = &fft_ctx[idx];

        /* Wait for buffer to be submitted */
        pthread_mutex_lock(&f->mutex);
        while (gpu_running && f->buffer_state != BUFFER_STATE_EXECUTING)
            pthread_cond_wait(&f->state_cond, &f->mutex);
        pthread_mutex_unlock(&f->mutex);
        if (!gpu_running) break;

        /* 1. Run PFB kernel via Metal compute */
        MTL::CommandBuffer *cmdBuf = f->queue->commandBuffer();
        if (!cmdBuf) {
            fprintf(stderr, "Metal: failed to create command buffer\n");
            break;
        }

        MTL::ComputeCommandEncoder *encoder = cmdBuf->computeCommandEncoder();
        if (!encoder) {
            fprintf(stderr, "Metal: failed to create compute encoder\n");
            break;
        }

        encoder->setComputePipelineState(pfb_pipeline);
        encoder->setBuffer(f->mtl_raw_buffer, 0, 0);
        encoder->setBuffer(mtl_h_sub, 0, 1);
        encoder->setBuffer(f->mtl_fft_buffer, 0, 2);
        encoder->setBuffer(param_M, 0, 3);
        encoder->setBuffer(param_M2, 0, 4);
        encoder->setBuffer(param_h_sub_len, 0, 5);
        encoder->setBuffer(param_pre_roll, 0, 6);
        encoder->setBuffer(param_batch_size, 0, 7);
        encoder->setBuffer(param_init_flag, 0, 8);

        /* Dispatch grid: [batch_size, M] threads */
        MTL::Size gridSize = MTL::Size(g_batch_size, pfb_M, 1);
        NS::UInteger w = pfb_pipeline->threadExecutionWidth();
        NS::UInteger h = pfb_pipeline->maxTotalThreadsPerThreadgroup() / w;
        if (h < 1) h = 1;
        MTL::Size threadgroupSize = MTL::Size(w, h, 1);

        encoder->dispatchThreads(gridSize, threadgroupSize);
        encoder->endEncoding();

        /* 2. Run VkFFT inverse FFT */
        VkFFTLaunchParams params = {};
        params.commandBuffer = cmdBuf;

        MTL::ComputeCommandEncoder *fftEncoder = cmdBuf->computeCommandEncoder();
        params.commandEncoder = fftEncoder;

        VkFFTResult res = VkFFTAppend(&f->app, 1, &params);
        if (res != VKFFT_SUCCESS) {
            fprintf(stderr, "Metal: VkFFT error %d\n", res);
            fftEncoder->endEncoding();
            break;
        }
        fftEncoder->endEncoding();

        /* Commit and wait for GPU completion */
        cmdBuf->commit();
        cmdBuf->waitUntilCompleted();

        /* 3. Read results from shared memory buffer */
        float *fft_output = (float *)f->mtl_fft_buffer->contents();

        /* 4. Deliver result to main thread */
        pthread_mutex_lock(&result_mutex);
        while (gpu_running && result_ptr != nullptr)
            pthread_cond_wait(&result_cond, &result_mutex);
        if (!gpu_running) {
            pthread_mutex_unlock(&result_mutex);
            break;
        }
        result_ptr = fft_output;
        result_ctx = f;
        pthread_cond_signal(&result_cond);
        pthread_mutex_unlock(&result_mutex);

        idx = (idx + 1) % NUM_FFT;
    }
    return nullptr;
}

/* ------------------------------------------------------------------ */
/* Internal helpers                                                    */
/* ------------------------------------------------------------------ */
static MTL::Buffer *make_param_buffer(unsigned value) {
    MTL::Buffer *buf = mtl_device->newBuffer(sizeof(unsigned),
                                              MTL::ResourceStorageModeShared);
    *(unsigned *)buf->contents() = value;
    return buf;
}

static int compile_pfb_kernel(void) {
    NS::Error *error = nullptr;

    NS::String *source = NS::String::string(pfb_kernel_source,
                                             NS::UTF8StringEncoding);
    MTL::CompileOptions *opts = MTL::CompileOptions::alloc()->init();

    MTL::Library *lib = mtl_device->newLibrary(source, opts, &error);
    opts->release();

    if (!lib) {
        if (error) {
            NS::String *desc = error->localizedDescription();
            fprintf(stderr, "Metal: shader compile error: %s\n",
                    desc->utf8String());
        }
        return -1;
    }

    NS::String *funcName = NS::String::string("pfb_channelize",
                                                NS::UTF8StringEncoding);
    MTL::Function *func = lib->newFunction(funcName);
    if (!func) {
        fprintf(stderr, "Metal: pfb_channelize function not found\n");
        lib->release();
        return -1;
    }

    pfb_pipeline = mtl_device->newComputePipelineState(func, &error);
    func->release();
    lib->release();

    if (!pfb_pipeline) {
        if (error) {
            NS::String *desc = error->localizedDescription();
            fprintf(stderr, "Metal: pipeline error: %s\n",
                    desc->utf8String());
        }
        return -1;
    }

    return 0;
}

static int init_fft_context(fft_context_t *f, unsigned width,
                            unsigned batch_size) {
    pthread_mutex_init(&f->mutex, nullptr);
    pthread_cond_init(&f->state_cond, nullptr);
    f->buffer_state = BUFFER_STATE_READY;

    f->queue = mtl_device->newCommandQueue();
    if (!f->queue) return -1;

    /* FFT buffer: float complex [batch_size * width], shared memory */
    f->fft_buffer_size = (uint64_t)sizeof(float) * 2 * width * batch_size;
    f->mtl_fft_buffer = mtl_device->newBuffer(f->fft_buffer_size,
                                               MTL::ResourceStorageModeShared);
    if (!f->mtl_fft_buffer) return -1;

    /* Raw buffer: int8 [(pre_roll_steps + batch_size) * M], shared memory */
    f->raw_buffer_size = (uint64_t)(pre_roll_steps + batch_size) * pfb_M;
    f->mtl_raw_buffer = mtl_device->newBuffer(f->raw_buffer_size,
                                               MTL::ResourceStorageModeShared);
    if (!f->mtl_raw_buffer) return -1;

    /* Zero-initialize raw buffer */
    memset(f->mtl_raw_buffer->contents(), 0, f->raw_buffer_size);

    /* VkFFT: 1D inverse FFT, batched, Metal backend */
    VkFFTConfiguration config = {};
    config.FFTdim = 1;
    config.size[0] = width;
    config.numberBatches = batch_size;
    config.device = mtl_device;
    config.queue = f->queue;
    config.buffer = &f->mtl_fft_buffer;
    config.bufferSize = &f->fft_buffer_size;

    VkFFTResult r = initializeVkFFT(&f->app, config);
    if (r != VKFFT_SUCCESS) {
        fprintf(stderr, "Metal: VkFFT init error %d\n", r);
        return -1;
    }

    return 0;
}

static void submit_current_buffer(void) {
    fft_context_t *f = &fft_ctx[cur_fft];
    if (f->buffer_state != BUFFER_STATE_FILLING)
        return;

    pthread_mutex_lock(&f->mutex);
    f->buffer_state = BUFFER_STATE_EXECUTING;
    pthread_cond_signal(&f->state_cond);
    pthread_mutex_unlock(&f->mutex);
}

static float *wait_for_result(void) {
    float *r;
    fft_context_t *ctx;

    pthread_mutex_lock(&result_mutex);
    while (gpu_running && result_ptr == nullptr)
        pthread_cond_wait(&result_cond, &result_mutex);
    r = result_ptr;
    ctx = result_ctx;
    result_ptr = nullptr;
    result_ctx = nullptr;
    pthread_cond_signal(&result_cond);
    pthread_mutex_unlock(&result_mutex);

    if (ctx) {
        pthread_mutex_lock(&ctx->mutex);
        ctx->buffer_state = BUFFER_STATE_READY;
        pthread_cond_signal(&ctx->state_cond);
        pthread_mutex_unlock(&ctx->mutex);
    }

    return r;
}

/* ------------------------------------------------------------------ */
/* Public API (extern "C" -- called from Rust via FFI)                 */
/* ------------------------------------------------------------------ */
extern "C" {

int gpu_pfb_init(unsigned M, unsigned h_sub_len,
                 const int16_t *h_sub, unsigned batch_size) {
    pfb_M = M;
    pfb_M2 = M / 2;
    pfb_h_sub_len = h_sub_len;
    g_batch_size = batch_size;

    pre_roll_steps = 2 * h_sub_len - 1;
    pre_roll_bytes = pre_roll_steps * pfb_M;

    pre_roll_buf = (int8_t *)calloc(1, pre_roll_bytes);
    if (!pre_roll_buf) return -1;

    /* Find Metal device */
    mtl_device = MTL::CreateSystemDefaultDevice();
    if (!mtl_device) {
        fprintf(stderr, "Metal: no GPU device found\n");
        return -1;
    }

    NS::String *devName = mtl_device->name();
    fprintf(stderr, "Metal GPU: %s\n", devName->utf8String());

    /* Compile PFB compute shader from MSL source */
    if (compile_pfb_kernel() != 0)
        return -1;

    /* Upload PFB coefficients (int16 -> float, normalized) */
    unsigned coeff_count = M * h_sub_len;
    mtl_h_sub = mtl_device->newBuffer(coeff_count * sizeof(float),
                                       MTL::ResourceStorageModeShared);
    if (!mtl_h_sub) return -1;

    float *h_sub_float = (float *)mtl_h_sub->contents();
    for (unsigned i = 0; i < coeff_count; i++)
        h_sub_float[i] = (float)h_sub[i] / 32768.0f;

    /* Create parameter buffers for shader constants */
    param_M = make_param_buffer(pfb_M);
    param_M2 = make_param_buffer(pfb_M2);
    param_h_sub_len = make_param_buffer(pfb_h_sub_len);
    param_pre_roll = make_param_buffer(pre_roll_steps);
    param_batch_size = make_param_buffer(g_batch_size);
    param_init_flag = make_param_buffer(1);

    /* Init double-buffered FFT contexts */
    for (unsigned i = 0; i < NUM_FFT; i++) {
        if (init_fft_context(&fft_ctx[i], M, batch_size) != 0)
            return -1;
    }

    /* Start GPU worker thread */
    gpu_running = 1;
    first_submit = 1;
    cur_fft = 0;
    result_ptr = nullptr;
    result_ctx = nullptr;

    if (pthread_create(&worker_thread, nullptr, fft_worker, nullptr) != 0) {
        fprintf(stderr, "Metal: failed to create worker thread\n");
        return -1;
    }

    /* Set up first buffer for filling */
    fft_ctx[0].buffer_state = BUFFER_STATE_FILLING;
    memset(fft_ctx[0].mtl_raw_buffer->contents(), 0, pre_roll_bytes);

    fprintf(stderr, "Metal GPU: M=%u h_sub_len=%u batch=%u pre_roll=%u\n",
            pfb_M, pfb_h_sub_len, g_batch_size, pre_roll_steps);

    return 0;
}

int8_t *gpu_pfb_get_buffer(void) {
    fft_context_t *f = &fft_ctx[cur_fft];
    int8_t *raw_ptr = (int8_t *)f->mtl_raw_buffer->contents();
    return &raw_ptr[pre_roll_bytes];
}

unsigned gpu_pfb_buffer_len(void) {
    return g_batch_size * pfb_M;
}

float *gpu_pfb_submit(void) {
    fft_context_t *f = &fft_ctx[cur_fft];
    float *prev_result = nullptr;

    /* Save tail of current buffer as pre-roll for next batch */
    if (f->buffer_state == BUFFER_STATE_FILLING) {
        int8_t *raw_ptr = (int8_t *)f->mtl_raw_buffer->contents();
        size_t batch_tail = pre_roll_bytes +
                            (size_t)(g_batch_size - pre_roll_steps) * pfb_M;
        memcpy(pre_roll_buf, &raw_ptr[batch_tail], pre_roll_bytes);
    }

    submit_current_buffer();

    if (!first_submit) {
        prev_result = wait_for_result();
    }
    first_submit = 0;

    cur_fft = (cur_fft + 1) % NUM_FFT;
    f = &fft_ctx[cur_fft];

    pthread_mutex_lock(&f->mutex);
    while (gpu_running && f->buffer_state != BUFFER_STATE_READY)
        pthread_cond_wait(&f->state_cond, &f->mutex);
    f->buffer_state = BUFFER_STATE_FILLING;
    pthread_mutex_unlock(&f->mutex);

    /* Copy pre-roll to beginning of next buffer */
    int8_t *raw_ptr = (int8_t *)f->mtl_raw_buffer->contents();
    memcpy(raw_ptr, pre_roll_buf, pre_roll_bytes);

    return prev_result;
}

float *gpu_pfb_flush(void) {
    return wait_for_result();
}

unsigned gpu_pfb_result_len(void) {
    return g_batch_size * pfb_M * 2;
}

unsigned gpu_pfb_batch_size(void) {
    return g_batch_size;
}

unsigned gpu_pfb_num_channels(void) {
    return pfb_M;
}

void gpu_pfb_deinit(void) {
    gpu_running = 0;

    /* Wake up worker thread */
    for (unsigned i = 0; i < NUM_FFT; i++) {
        pthread_mutex_lock(&fft_ctx[i].mutex);
        pthread_cond_signal(&fft_ctx[i].state_cond);
        pthread_mutex_unlock(&fft_ctx[i].mutex);
    }
    pthread_mutex_lock(&result_mutex);
    pthread_cond_signal(&result_cond);
    pthread_mutex_unlock(&result_mutex);

    /* Brief wait, then cancel if stuck */
    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = 100000000; /* 100ms */
    nanosleep(&ts, nullptr);

    pthread_cancel(worker_thread);
    pthread_detach(worker_thread);

    free(pre_roll_buf);
    pre_roll_buf = nullptr;

    /* Release Metal objects */
    for (unsigned i = 0; i < NUM_FFT; i++) {
        deleteVkFFT(&fft_ctx[i].app);
        if (fft_ctx[i].mtl_fft_buffer) fft_ctx[i].mtl_fft_buffer->release();
        if (fft_ctx[i].mtl_raw_buffer) fft_ctx[i].mtl_raw_buffer->release();
        if (fft_ctx[i].queue) fft_ctx[i].queue->release();
        pthread_mutex_destroy(&fft_ctx[i].mutex);
        pthread_cond_destroy(&fft_ctx[i].state_cond);
    }

    if (pfb_pipeline) pfb_pipeline->release();
    if (mtl_h_sub) mtl_h_sub->release();
    if (param_M) param_M->release();
    if (param_M2) param_M2->release();
    if (param_h_sub_len) param_h_sub_len->release();
    if (param_pre_roll) param_pre_roll->release();
    if (param_batch_size) param_batch_size->release();
    if (param_init_flag) param_init_flag->release();
    if (mtl_device) mtl_device->release();
}

} /* extern "C" */
