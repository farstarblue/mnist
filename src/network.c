#include "network.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CONV_PATCH_SIZE (CONV_KERNEL_SIZE * CONV_KERNEL_SIZE)
#define CONV_WEIGHT_COUNT (CONV_OUT_CHANNELS * CONV_PATCH_SIZE)
#define FC_WEIGHT_COUNT (FEATURE_SIZE * OUTPUT_SIZE)

static float uniform_random(void) {
    return ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
}

static float normal_random(void) {
    float u1 = uniform_random();
    float u2 = uniform_random();
    float radius = sqrtf(-2.0f * logf(u1));
    float angle = 2.0f * (float)M_PI * u2;
    return radius * cosf(angle);
}

static float relu(float value) {
    return value > 0.0f ? value : 0.0f;
}

#if defined(__riscv_vector)
static float rvv_dot_vectorized(const float *lhs, const float *rhs, int count) {
    float sum = 0.0f;
    int index = 0;

    while (index < count) {
        size_t vl = __riscv_vsetvl_e32m4((size_t)(count - index));
        vfloat32m4_t lhs_vector = __riscv_vle32_v_f32m4(lhs + index, vl);
        vfloat32m4_t rhs_vector = __riscv_vle32_v_f32m4(rhs + index, vl);
        vfloat32m4_t products = __riscv_vfmul_vv_f32m4(lhs_vector, rhs_vector, vl);
        vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t reduced = __riscv_vfredusum_vs_f32m4_f32m1(products, zero, vl);
        sum += __riscv_vfmv_f_s_f32m1_f32(reduced);
        index += (int)vl;
    }

    return sum;
}

static void axpy_inplace_rvv(float *dst, const float *src, float alpha, int count) {
    int index = 0;

    while (index < count) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)(count - index));
        vfloat32m1_t dst_vector = __riscv_vle32_v_f32m1(dst + index, vl);
        vfloat32m1_t src_vector = __riscv_vle32_v_f32m1(src + index, vl);
        dst_vector = __riscv_vfmacc_vf_f32m1(dst_vector, alpha, src_vector, vl);
        __riscv_vse32_v_f32m1(dst + index, dst_vector, vl);
        index += (int)vl;
    }
}

static float conv_dot_rvv(const float *input, int top, int left, const float *kernel) {
    float patch[CONV_PATCH_SIZE];
    int index = 0;
    int kh = 0;
    int kw = 0;

    for (kh = 0; kh < CONV_KERNEL_SIZE; ++kh) {
        for (kw = 0; kw < CONV_KERNEL_SIZE; ++kw) {
            patch[index++] = input[(top + kh) * MNIST_COLS + (left + kw)];
        }
    }

    return rvv_dot_vectorized(kernel, patch, CONV_PATCH_SIZE);
}
#endif

static float dot_product(const float *lhs, const float *rhs, int count) {
    int index = 0;
    float sum = 0.0f;

#if defined(__riscv_vector)
    return rvv_dot_vectorized(lhs, rhs, count);
#endif

    for (index = 0; index < count; ++index) {
        sum += lhs[index] * rhs[index];
    }

    return sum;
}

static void axpy_inplace(float *dst, const float *src, float alpha, int count) {
    int index = 0;

#if defined(__riscv_vector)
    axpy_inplace_rvv(dst, src, alpha, count);
    return;
#endif

    for (index = 0; index < count; ++index) {
        dst[index] += alpha * src[index];
    }
}

#if !defined(__riscv_vector)
static float conv_dot_scalar(const float *input, int top, int left, const float *kernel) {
    float sum = 0.0f;
    int kh = 0;
    int kw = 0;

    for (kh = 0; kh < CONV_KERNEL_SIZE; ++kh) {
        for (kw = 0; kw < CONV_KERNEL_SIZE; ++kw) {
            sum += input[(top + kh) * MNIST_COLS + (left + kw)] * kernel[kh * CONV_KERNEL_SIZE + kw];
        }
    }

    return sum;
}
#endif

static float conv_dot(const float *input, int top, int left, const float *kernel) {
#if defined(__riscv_vector)
    return conv_dot_rvv(input, top, left, kernel);
#else
    return conv_dot_scalar(input, top, left, kernel);
#endif
}

static void softmax(const float *logits, float *probabilities) {
    float max_value = logits[0];
    float sum = 0.0f;
    int i = 0;

    for (i = 1; i < OUTPUT_SIZE; ++i) {
        if (logits[i] > max_value) {
            max_value = logits[i];
        }
    }

    for (i = 0; i < OUTPUT_SIZE; ++i) {
        probabilities[i] = expf(logits[i] - max_value);
        sum += probabilities[i];
    }

    for (i = 0; i < OUTPUT_SIZE; ++i) {
        probabilities[i] /= sum;
    }
}

static void conv_forward(const Network *network, const float *input, float *features) {
    int out_channel = 0;

    for (out_channel = 0; out_channel < CONV_OUT_CHANNELS; ++out_channel) {
        const float *kernel = network->conv_w + (size_t)out_channel * CONV_PATCH_SIZE;
        float bias = network->conv_b[out_channel];
        int out_row = 0;
        int out_col = 0;

        for (out_row = 0; out_row < CONV_OUT_ROWS; ++out_row) {
            for (out_col = 0; out_col < CONV_OUT_COLS; ++out_col) {
                size_t index = (size_t)out_channel * CONV_OUT_ROWS * CONV_OUT_COLS +
                               (size_t)out_row * CONV_OUT_COLS +
                               (size_t)out_col;
                float sum = bias + conv_dot(input, out_row, out_col, kernel);
                features[index] = relu(sum);
            }
        }
    }
}

int network_init(Network *network) {
    memset(network, 0, sizeof(*network));

    network->conv_w = (float *)malloc(CONV_WEIGHT_COUNT * sizeof(float));
    network->conv_b = (float *)calloc(CONV_OUT_CHANNELS, sizeof(float));
    network->fc_w = (float *)malloc(FC_WEIGHT_COUNT * sizeof(float));
    network->fc_b = (float *)calloc(OUTPUT_SIZE, sizeof(float));
    network->dconv_w = (float *)calloc(CONV_WEIGHT_COUNT, sizeof(float));
    network->dconv_b = (float *)calloc(CONV_OUT_CHANNELS, sizeof(float));
    network->dfc_w = (float *)calloc(FC_WEIGHT_COUNT, sizeof(float));
    network->dfc_b = (float *)calloc(OUTPUT_SIZE, sizeof(float));

    if (network->conv_w == NULL || network->conv_b == NULL || network->fc_w == NULL ||
        network->fc_b == NULL || network->dconv_w == NULL || network->dconv_b == NULL ||
        network->dfc_w == NULL || network->dfc_b == NULL) {
        fprintf(stderr, "网络内存分配失败。\n");
        network_free(network);
        return -1;
    }

    return 0;
}

void network_free(Network *network) {
    free(network->conv_w);
    free(network->conv_b);
    free(network->fc_w);
    free(network->fc_b);
    free(network->dconv_w);
    free(network->dconv_b);
    free(network->dfc_w);
    free(network->dfc_b);
    memset(network, 0, sizeof(*network));
}

void network_he_init(Network *network) {
    float conv_scale = sqrtf(2.0f / (float)CONV_PATCH_SIZE);
    float fc_scale = sqrtf(2.0f / (float)FEATURE_SIZE);
    size_t index = 0;

    for (index = 0; index < CONV_WEIGHT_COUNT; ++index) {
        network->conv_w[index] = normal_random() * conv_scale;
    }

    for (index = 0; index < FC_WEIGHT_COUNT; ++index) {
        network->fc_w[index] = normal_random() * fc_scale;
    }

    memset(network->conv_b, 0, CONV_OUT_CHANNELS * sizeof(float));
    memset(network->fc_b, 0, OUTPUT_SIZE * sizeof(float));
}

void network_zero_grad(Network *network) {
    memset(network->dconv_w, 0, CONV_WEIGHT_COUNT * sizeof(float));
    memset(network->dconv_b, 0, CONV_OUT_CHANNELS * sizeof(float));
    memset(network->dfc_w, 0, FC_WEIGHT_COUNT * sizeof(float));
    memset(network->dfc_b, 0, OUTPUT_SIZE * sizeof(float));
}

void network_forward(const Network *network,
                     const float *input,
                     float *features,
                     float *logits,
                     float *probabilities) {
    int output_index = 0;

    conv_forward(network, input, features);

    for (output_index = 0; output_index < OUTPUT_SIZE; ++output_index) {
        float sum = network->fc_b[output_index];
        const float *weights = network->fc_w + (size_t)output_index * FEATURE_SIZE;
        sum += dot_product(weights, features, FEATURE_SIZE);
        logits[output_index] = sum;
    }

    softmax(logits, probabilities);
}

float network_accumulate_gradients(Network *network,
                                   const float *input,
                                   unsigned char label,
                                   const float *features,
                                   const float *probabilities) {
    float output_delta[OUTPUT_SIZE];
    float feature_delta[FEATURE_SIZE];
    float loss = 0.0f;
    int output_index = 0;
    int feature_index = 0;

    for (output_index = 0; output_index < OUTPUT_SIZE; ++output_index) {
        output_delta[output_index] = probabilities[output_index] - (output_index == (int)label ? 1.0f : 0.0f);
    }

    loss = -logf(probabilities[label] + 1e-7f);

    for (output_index = 0; output_index < OUTPUT_SIZE; ++output_index) {
        const float delta = output_delta[output_index];
        float *weights_grad = network->dfc_w + (size_t)output_index * FEATURE_SIZE;
        network->dfc_b[output_index] += delta;
        axpy_inplace(weights_grad, features, delta, FEATURE_SIZE);
    }

    for (feature_index = 0; feature_index < FEATURE_SIZE; ++feature_index) {
        float gradient = 0.0f;

        for (output_index = 0; output_index < OUTPUT_SIZE; ++output_index) {
            gradient += output_delta[output_index] * network->fc_w[(size_t)output_index * FEATURE_SIZE + feature_index];
        }

        feature_delta[feature_index] = features[feature_index] > 0.0f ? gradient : 0.0f;
    }

    for (feature_index = 0; feature_index < FEATURE_SIZE; ++feature_index) {
        int out_channel = feature_index / (CONV_OUT_ROWS * CONV_OUT_COLS);
        int spatial_index = feature_index % (CONV_OUT_ROWS * CONV_OUT_COLS);
        int out_row = spatial_index / CONV_OUT_COLS;
        int out_col = spatial_index % CONV_OUT_COLS;
        float delta = feature_delta[feature_index];
        float *kernel_grad = network->dconv_w + (size_t)out_channel * CONV_PATCH_SIZE;
        int kh = 0;
        int kw = 0;

        network->dconv_b[out_channel] += delta;
        for (kh = 0; kh < CONV_KERNEL_SIZE; ++kh) {
            for (kw = 0; kw < CONV_KERNEL_SIZE; ++kw) {
                kernel_grad[kh * CONV_KERNEL_SIZE + kw] +=
                    delta * input[(out_row + kh) * MNIST_COLS + (out_col + kw)];
            }
        }
    }

    return loss;
}

void network_apply_gradients(Network *network, float learning_rate, int batch_size) {
    float scale = learning_rate / (float)batch_size;

    axpy_inplace(network->conv_w, network->dconv_w, -scale, CONV_WEIGHT_COUNT);
    axpy_inplace(network->conv_b, network->dconv_b, -scale, CONV_OUT_CHANNELS);
    axpy_inplace(network->fc_w, network->dfc_w, -scale, FC_WEIGHT_COUNT);
    axpy_inplace(network->fc_b, network->dfc_b, -scale, OUTPUT_SIZE);
}

int network_predict(const Network *network,
                    const float *input,
                    float *workspace_features,
                    float *workspace_logits,
                    float *workspace_probabilities) {
    int best_index = 0;
    int index = 0;

    network_forward(network, input, workspace_features, workspace_logits, workspace_probabilities);

    for (index = 1; index < OUTPUT_SIZE; ++index) {
        if (workspace_probabilities[index] > workspace_probabilities[best_index]) {
            best_index = index;
        }
    }

    return best_index;
}

float network_accuracy(const Network *network, const float *images, const unsigned char *labels, int count) {
    float features[FEATURE_SIZE];
    float logits[OUTPUT_SIZE];
    float probabilities[OUTPUT_SIZE];
    int correct = 0;
    int index = 0;

    for (index = 0; index < count; ++index) {
        int prediction = network_predict(network,
                                         images + (size_t)index * INPUT_SIZE,
                                         features,
                                         logits,
                                         probabilities);
        if (prediction == labels[index]) {
            ++correct;
        }
    }

    return (float)correct / (float)count;
}

void network_load_parameters(Network *network,
                             const float *conv_w,
                             const float *conv_b,
                             const float *fc_w,
                             const float *fc_b) {
    memcpy(network->conv_w, conv_w, CONV_WEIGHT_COUNT * sizeof(float));
    memcpy(network->conv_b, conv_b, CONV_OUT_CHANNELS * sizeof(float));
    memcpy(network->fc_w, fc_w, FC_WEIGHT_COUNT * sizeof(float));
    memcpy(network->fc_b, fc_b, OUTPUT_SIZE * sizeof(float));
}
