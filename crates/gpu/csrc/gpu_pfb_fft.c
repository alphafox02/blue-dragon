/*
 * GPU PFB + FFT for blue-dragon
 * OpenCL polyphase channelizer + VkFFT inverse FFT
 *
 * Double-buffered: GPU processes batch N while host fills batch N+1.
 *
 * Public API (called from Rust via FFI):
 *   gpu_pfb_init()        - init OpenCL, compile kernel, start worker
 *   gpu_pfb_get_buffer()  - get raw int8 buffer to fill
 *   gpu_pfb_submit()      - submit buffer, get previous result
 *   gpu_pfb_flush()       - get final result
 *   gpu_pfb_deinit()      - shutdown
 */

#define CL_TARGET_OPENCL_VERSION 120
#define VKFFT_BACKEND 3

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <CL/opencl.h>

#include "vkFFT.h"

#define NUM_FFT 2

/* ------------------------------------------------------------------ */
/* OpenCL PFB kernel source (int16 input)                              */
/* ------------------------------------------------------------------ */
static const char *pfb_kernel_source =
"__kernel void pfb_channelize(\n"
"    __global const short *raw_input,\n"
"    __global const float *h_sub,\n"
"    __global float *fft_buffer,\n"
"    const uint M,\n"
"    const uint M2,\n"
"    const uint h_sub_len,\n"
"    const uint pre_roll_steps,\n"
"    const uint batch_size,\n"
"    const uint initial_flag\n"
") {\n"
"    uint t = get_global_id(0);\n"
"    uint ch = get_global_id(1);\n"
"    if (t >= batch_size || ch >= M) return;\n"
"\n"
"    uint abs_t = t + pre_roll_steps;\n"
"    uint flag_at_t = (initial_flag + abs_t) & 1u;\n"
"\n"
"    uint sub_offset = flag_at_t\n"
"        ? ((M2 + ch) >= M ? (M2 + ch - M) : (M2 + ch))\n"
"        : ch;\n"
"\n"
"    uint is_lower = (ch < M2) ? 1u : 0u;\n"
"    uint push_parity = is_lower\n"
"        ? (initial_flag & 1u)\n"
"        : ((initial_flag + 1u) & 1u);\n"
"\n"
"    uint latest_push = ((abs_t & 1u) == push_parity)\n"
"        ? abs_t : (abs_t - 1u);\n"
"\n"
"    uint ch_pos = is_lower\n"
"        ? (M2 - 1u - ch)\n"
"        : (M - 1u - ch);\n"
"\n"
"    float real_sum = 0.0f, imag_sum = 0.0f;\n"
"    for (uint k = 0; k < h_sub_len; ++k) {\n"
"        uint push_step = latest_push - 2u * (h_sub_len - 1u - k);\n"
"        uint raw_idx = push_step * M2 + ch_pos;\n"
"        float sr = (float)(raw_input[raw_idx * 2u]);\n"
"        float si = (float)(raw_input[raw_idx * 2u + 1u]);\n"
"        float coeff = h_sub[sub_offset * h_sub_len + k];\n"
"        real_sum += sr * coeff;\n"
"        imag_sum += si * coeff;\n"
"    }\n"
"\n"
"    float scale = 1.0f / 32768.0f;\n"
"    uint out_idx = t * M + ch;\n"
"    fft_buffer[out_idx * 2u] = real_sum * scale;\n"
"    fft_buffer[out_idx * 2u + 1u] = imag_sum * scale;\n"
"}\n";

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
    cl_command_queue queue;

    cl_mem cl_fft_buffer;
    cl_mem cl_raw_buffer;

    float *fft_host_ptr;
    int16_t *raw_host_ptr;

    uint64_t fft_buffer_size;
    uint64_t raw_buffer_size;

    buffer_state_t buffer_state;

    pthread_mutex_t mutex;
    pthread_cond_t state_cond;
} fft_context_t;

static fft_context_t fft_ctx[NUM_FFT];
static unsigned cur_fft = 0;

/* OpenCL globals */
static cl_context cl_ctx;
static cl_device_id cl_device;
static cl_program pfb_program;
static cl_kernel pfb_kern;
static cl_mem cl_h_sub;

/* PFB parameters */
static unsigned pfb_M, pfb_M2, pfb_h_sub_len;
static unsigned g_batch_size;
static unsigned pre_roll_steps;
static unsigned pre_roll_bytes;
static int16_t *pre_roll_buf;

/* Worker thread */
static pthread_t worker_thread;
static volatile int gpu_running = 0;

/* Result delivery: GPU worker writes result, main thread reads it */
static float *result_ptr = NULL;
static fft_context_t *result_ctx = NULL;
static pthread_mutex_t result_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t result_cond = PTHREAD_COND_INITIALIZER;

/* Track whether first batch has been submitted */
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

        /* 1. Upload raw int8 data to GPU */
        cl_int err = clEnqueueWriteBuffer(f->queue, f->cl_raw_buffer,
                                          CL_TRUE, 0, f->raw_buffer_size,
                                          f->raw_host_ptr, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "GPU: raw upload error %d\n", err);
            break;
        }

        /* 2. Launch PFB kernel */
        size_t global_work[2] = { g_batch_size, pfb_M };
        clSetKernelArg(pfb_kern, 0, sizeof(cl_mem), &f->cl_raw_buffer);
        clSetKernelArg(pfb_kern, 1, sizeof(cl_mem), &cl_h_sub);
        clSetKernelArg(pfb_kern, 2, sizeof(cl_mem), &f->cl_fft_buffer);
        clSetKernelArg(pfb_kern, 3, sizeof(cl_uint), &pfb_M);
        clSetKernelArg(pfb_kern, 4, sizeof(cl_uint), &pfb_M2);
        clSetKernelArg(pfb_kern, 5, sizeof(cl_uint), &pfb_h_sub_len);
        clSetKernelArg(pfb_kern, 6, sizeof(cl_uint), &pre_roll_steps);
        cl_uint bs = g_batch_size;
        clSetKernelArg(pfb_kern, 7, sizeof(cl_uint), &bs);
        cl_uint init_flag = 1;
        clSetKernelArg(pfb_kern, 8, sizeof(cl_uint), &init_flag);

        err = clEnqueueNDRangeKernel(f->queue, pfb_kern, 2, NULL,
                                     global_work, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "GPU: PFB kernel error %d\n", err);
            break;
        }

        /* 3. Inverse FFT via VkFFT */
        VkFFTLaunchParams params = {};
        params.commandQueue = &f->queue;
        VkFFTResult res = VkFFTAppend(&f->app, 1, &params);
        if (res != VKFFT_SUCCESS) {
            fprintf(stderr, "GPU: VkFFT error %d\n", res);
            break;
        }
        clFinish(f->queue);

        /* 4. Readback FFT output to host */
        err = clEnqueueReadBuffer(f->queue, f->cl_fft_buffer, CL_TRUE,
                                  0, f->fft_buffer_size, f->fft_host_ptr,
                                  0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "GPU: readback error %d\n", err);
            break;
        }

        /* 5. Deliver result to main thread */
        pthread_mutex_lock(&result_mutex);
        /* Wait for previous result to be consumed */
        while (gpu_running && result_ptr != NULL)
            pthread_cond_wait(&result_cond, &result_mutex);
        if (!gpu_running) {
            pthread_mutex_unlock(&result_mutex);
            break;
        }
        result_ptr = f->fft_host_ptr;
        result_ctx = f;
        pthread_cond_signal(&result_cond);
        pthread_mutex_unlock(&result_mutex);

        idx = (idx + 1) % NUM_FFT;
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Internal helpers                                                    */
/* ------------------------------------------------------------------ */
static cl_device_id find_opencl_device(char *name_out, size_t name_len) {
    cl_int err;
    cl_uint num_platforms;
    cl_platform_id platforms[16];
    cl_device_id device = 0;

    err = clGetPlatformIDs(16, platforms, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0)
        return 0;

    /* Prefer GPU */
    for (cl_uint p = 0; p < num_platforms; p++) {
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        if (err == CL_SUCCESS) {
            if (name_out)
                clGetDeviceInfo(device, CL_DEVICE_NAME, name_len, name_out, NULL);
            return device;
        }
    }

    /* Fall back to any device */
    for (cl_uint p = 0; p < num_platforms; p++) {
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 1, &device, NULL);
        if (err == CL_SUCCESS) {
            if (name_out)
                clGetDeviceInfo(device, CL_DEVICE_NAME, name_len, name_out, NULL);
            return device;
        }
    }

    return 0;
}

static int init_fft_context(fft_context_t *f, unsigned width,
                            unsigned batch_size) {
    cl_int err;

    pthread_mutex_init(&f->mutex, NULL);
    pthread_cond_init(&f->state_cond, NULL);
    f->buffer_state = BUFFER_STATE_READY;

    f->queue = clCreateCommandQueue(cl_ctx, cl_device, 0, &err);
    if (err != CL_SUCCESS) return -1;

    /* FFT buffer: float complex [batch_size * width] */
    f->fft_buffer_size = (uint64_t)sizeof(float) * 2 * width * batch_size;
    f->fft_host_ptr = (float *)malloc(f->fft_buffer_size);
    if (!f->fft_host_ptr) return -1;

    f->cl_fft_buffer = clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE,
                                      f->fft_buffer_size, NULL, &err);
    if (err != CL_SUCCESS) return -1;

    /* Raw buffer: int16 [(pre_roll_steps + batch_size) * M] */
    f->raw_buffer_size = (uint64_t)(pre_roll_steps + batch_size) * pfb_M * sizeof(int16_t);
    f->raw_host_ptr = (int16_t *)calloc(1, f->raw_buffer_size);
    if (!f->raw_host_ptr) return -1;

    f->cl_raw_buffer = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY,
                                      f->raw_buffer_size, NULL, &err);
    if (err != CL_SUCCESS) return -1;

    /* VkFFT: 1D inverse FFT, batched */
    VkFFTConfiguration config = {};
    config.FFTdim = 1;
    config.size[0] = width;
    config.numberBatches = batch_size;
    config.device = &cl_device;
    config.context = &cl_ctx;
    config.commandQueue = &f->queue;
    config.buffer = &f->cl_fft_buffer;
    config.bufferSize = &f->fft_buffer_size;

    VkFFTResult r = initializeVkFFT(&f->app, config);
    if (r != VKFFT_SUCCESS) {
        fprintf(stderr, "GPU: VkFFT init error %d\n", r);
        return -1;
    }

    return 0;
}

static int compile_pfb_kernel(void) {
    cl_int err;

    pfb_program = clCreateProgramWithSource(cl_ctx, 1, &pfb_kernel_source,
                                            NULL, &err);
    if (err != CL_SUCCESS) return -1;

    err = clBuildProgram(pfb_program, 1, &cl_device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(pfb_program, cl_device, CL_PROGRAM_BUILD_LOG,
                              sizeof(log), log, NULL);
        fprintf(stderr, "GPU: kernel build error: %s\n", log);
        return -1;
    }

    pfb_kern = clCreateKernel(pfb_program, "pfb_channelize", &err);
    if (err != CL_SUCCESS) return -1;

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

/* Wait for result from GPU worker. Returns float* or NULL. */
static float *wait_for_result(void) {
    float *r;
    fft_context_t *ctx;

    pthread_mutex_lock(&result_mutex);
    while (gpu_running && result_ptr == NULL)
        pthread_cond_wait(&result_cond, &result_mutex);
    r = result_ptr;
    ctx = result_ctx;
    result_ptr = NULL;
    result_ctx = NULL;
    /* Signal worker that result was consumed */
    pthread_cond_signal(&result_cond);
    pthread_mutex_unlock(&result_mutex);

    /* Release the buffer context back to READY */
    if (ctx) {
        pthread_mutex_lock(&ctx->mutex);
        ctx->buffer_state = BUFFER_STATE_READY;
        pthread_cond_signal(&ctx->state_cond);
        pthread_mutex_unlock(&ctx->mutex);
    }

    return r;
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

int gpu_pfb_init(unsigned M, unsigned h_sub_len,
                 const int16_t *h_sub, unsigned batch_size) {
    char dev_name[256] = {0};

    pfb_M = M;
    pfb_M2 = M / 2;
    pfb_h_sub_len = h_sub_len;
    g_batch_size = batch_size;

    pre_roll_steps = 2 * h_sub_len - 1;
    pre_roll_bytes = pre_roll_steps * pfb_M * sizeof(int16_t);

    pre_roll_buf = (int16_t *)calloc(1, pre_roll_bytes);
    if (!pre_roll_buf) return -1;

    /* Find OpenCL device */
    cl_device = find_opencl_device(dev_name, sizeof(dev_name));
    if (!cl_device) {
        fprintf(stderr, "GPU: no OpenCL device found\n");
        return -1;
    }
    fprintf(stderr, "GPU: %s\n", dev_name);

    cl_int err;
    cl_ctx = clCreateContext(NULL, 1, &cl_device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "GPU: context creation error %d\n", err);
        return -1;
    }

    /* Upload PFB coefficients (int16 -> float, normalized) */
    unsigned coeff_count = M * h_sub_len;
    float *h_sub_float = (float *)malloc(coeff_count * sizeof(float));
    if (!h_sub_float) return -1;
    for (unsigned i = 0; i < coeff_count; i++)
        h_sub_float[i] = (float)h_sub[i] / 32768.0f;

    cl_h_sub = clCreateBuffer(cl_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              coeff_count * sizeof(float), h_sub_float, &err);
    free(h_sub_float);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "GPU: coefficient buffer error %d\n", err);
        return -1;
    }

    /* Compile PFB kernel */
    if (compile_pfb_kernel() != 0)
        return -1;

    /* Init double-buffered FFT contexts */
    for (unsigned i = 0; i < NUM_FFT; i++) {
        if (init_fft_context(&fft_ctx[i], M, batch_size) != 0)
            return -1;
    }

    /* Start GPU worker thread */
    gpu_running = 1;
    first_submit = 1;
    cur_fft = 0;
    result_ptr = NULL;
    result_ctx = NULL;

    if (pthread_create(&worker_thread, NULL, fft_worker, NULL) != 0) {
        fprintf(stderr, "GPU: failed to create worker thread\n");
        return -1;
    }

    /* Set up first buffer for filling */
    fft_ctx[0].buffer_state = BUFFER_STATE_FILLING;
    memset(fft_ctx[0].raw_host_ptr, 0, pre_roll_bytes);

    fprintf(stderr, "GPU: M=%u h_sub_len=%u batch=%u pre_roll=%u\n",
            pfb_M, pfb_h_sub_len, g_batch_size, pre_roll_steps);

    return 0;
}

int16_t *gpu_pfb_get_buffer(void) {
    fft_context_t *f = &fft_ctx[cur_fft];
    /* Return pointer to batch area (after pre-roll, in int16 elements) */
    unsigned pre_roll_elements = pre_roll_steps * pfb_M;
    return &f->raw_host_ptr[pre_roll_elements];
}

unsigned gpu_pfb_buffer_len(void) {
    /* Returns number of int16 elements (not bytes) */
    return g_batch_size * pfb_M;
}

float *gpu_pfb_submit(void) {
    fft_context_t *f = &fft_ctx[cur_fft];
    float *prev_result = NULL;

    /* Save tail of current buffer as pre-roll for next batch */
    if (f->buffer_state == BUFFER_STATE_FILLING) {
        unsigned pre_roll_elements = pre_roll_steps * pfb_M;
        size_t batch_tail = pre_roll_elements +
                            (size_t)(g_batch_size - pre_roll_steps) * pfb_M;
        memcpy(pre_roll_buf, &f->raw_host_ptr[batch_tail], pre_roll_bytes);
    }

    /* Submit current buffer to GPU worker */
    submit_current_buffer();

    /* If not the first submit, wait for previous batch result */
    if (!first_submit) {
        prev_result = wait_for_result();
    }
    first_submit = 0;

    /* Switch to next buffer */
    cur_fft = (cur_fft + 1) % NUM_FFT;
    f = &fft_ctx[cur_fft];

    /* Wait for next buffer to become available */
    pthread_mutex_lock(&f->mutex);
    while (gpu_running && f->buffer_state != BUFFER_STATE_READY)
        pthread_cond_wait(&f->state_cond, &f->mutex);
    f->buffer_state = BUFFER_STATE_FILLING;
    pthread_mutex_unlock(&f->mutex);

    /* Copy pre-roll to beginning of next buffer */
    memcpy(f->raw_host_ptr, pre_roll_buf, pre_roll_bytes);

    return prev_result;
}

float *gpu_pfb_flush(void) {
    /* Get the last submitted batch's result */
    return wait_for_result();
}

unsigned gpu_pfb_result_len(void) {
    /* Number of floats in a result: batch_size * M * 2 (interleaved complex) */
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

    /* Wake up worker thread if blocked */
    for (unsigned i = 0; i < NUM_FFT; i++) {
        pthread_mutex_lock(&fft_ctx[i].mutex);
        pthread_cond_signal(&fft_ctx[i].state_cond);
        pthread_mutex_unlock(&fft_ctx[i].mutex);
    }
    pthread_mutex_lock(&result_mutex);
    pthread_cond_signal(&result_cond);
    pthread_mutex_unlock(&result_mutex);

    /* Give worker a moment to exit, then cancel if stuck in GPU call */
    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = 100000000; /* 100ms */
    nanosleep(&ts, NULL);

    pthread_cancel(worker_thread);
    pthread_detach(worker_thread);

    free(pre_roll_buf);
    pre_roll_buf = NULL;

    for (unsigned i = 0; i < NUM_FFT; i++) {
        free(fft_ctx[i].fft_host_ptr);
        fft_ctx[i].fft_host_ptr = NULL;
        free(fft_ctx[i].raw_host_ptr);
        fft_ctx[i].raw_host_ptr = NULL;
        pthread_mutex_destroy(&fft_ctx[i].mutex);
        pthread_cond_destroy(&fft_ctx[i].state_cond);
    }
}
