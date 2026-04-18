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
static float dot_product_rvv(const float *lhs, const float *rhs, int count) {
    int index = 0;
    size_t vl = 0;
    vfloat32m1_t accumulator = __riscv_vfmv_v_f_f32m1(0.0f, 1);

    for (index = 0; index < count; index += (int)vl) {
        vfloat32m1_t lhs_vector;
        vfloat32m1_t rhs_vector;

        vl = __riscv_vsetvl_e32m1((size_t)(count - index));
        lhs_vector = __riscv_vle32_v_f32m1(lhs + index, vl);
        rhs_vector = __riscv_vle32_v_f32m1(rhs + index, vl);
        accumulator = __riscv_vfmacc_vv_f32m1(accumulator, lhs_vector, rhs_vector, vl);
    }

    {
        vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        vfloat32m1_t reduced = __riscv_vfredusum_vs_f32m1_f32m1(accumulator, zero, 1);
        return __riscv_vfmv_f_s_f32m1_f32(reduced);
    }
}

static void axpy_inplace_rvv(float *dst, const float *src, float alpha, int count) {
    int index = 0;
    size_t vl = 0;

    for (index = 0; index < count; index += (int)vl) {
        vfloat32m1_t dst_vector;
        vfloat32m1_t src_vector;

        vl = __riscv_vsetvl_e32m1((size_t)(count - index));
        dst_vector = __riscv_vle32_v_f32m1(dst + index, vl);
        src_vector = __riscv_vle32_v_f32m1(src + index, vl);
        dst_vector = __riscv_vfmacc_vf_f32m1(dst_vector, alpha, src_vector, vl);
        __riscv_vse32_v_f32m1(dst + index, dst_vector, vl);
    }
}
#endif

static float dot_product(const float *lhs, const float *rhs, int count) {
    int index = 0;
    float sum = 0.0f;

#if defined(__riscv_vector)
    return dot_product_rvv(lhs, rhs, count);
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

int network_init(Network *network) {
    size_t w1_size = INPUT_SIZE * HIDDEN_SIZE;
    size_t w2_size = HIDDEN_SIZE * OUTPUT_SIZE;

    memset(network, 0, sizeof(*network));

    network->w1 = (float *)malloc(w1_size * sizeof(float));
    network->b1 = (float *)calloc(HIDDEN_SIZE, sizeof(float));
    network->w2 = (float *)malloc(w2_size * sizeof(float));
    network->b2 = (float *)calloc(OUTPUT_SIZE, sizeof(float));
    network->dw1 = (float *)calloc(w1_size, sizeof(float));
    network->db1 = (float *)calloc(HIDDEN_SIZE, sizeof(float));
    network->dw2 = (float *)calloc(w2_size, sizeof(float));
    network->db2 = (float *)calloc(OUTPUT_SIZE, sizeof(float));

    if (network->w1 == NULL || network->b1 == NULL || network->w2 == NULL ||
        network->b2 == NULL || network->dw1 == NULL || network->db1 == NULL ||
        network->dw2 == NULL || network->db2 == NULL) {
        fprintf(stderr, "网络内存分配失败。\n");
        network_free(network);
        return -1;
    }

    return 0;
}

void network_free(Network *network) {
    free(network->w1);
    free(network->b1);
    free(network->w2);
    free(network->b2);
    free(network->dw1);
    free(network->db1);
    free(network->dw2);
    free(network->db2);
    memset(network, 0, sizeof(*network));
}

void network_he_init(Network *network) {
    float scale1 = sqrtf(2.0f / INPUT_SIZE);
    float scale2 = sqrtf(2.0f / HIDDEN_SIZE);
    size_t i = 0;

    for (i = 0; i < INPUT_SIZE * HIDDEN_SIZE; ++i) {
        network->w1[i] = normal_random() * scale1;
    }

    for (i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; ++i) {
        network->w2[i] = normal_random() * scale2;
    }

    memset(network->b1, 0, HIDDEN_SIZE * sizeof(float));
    memset(network->b2, 0, OUTPUT_SIZE * sizeof(float));
}

void network_zero_grad(Network *network) {
    memset(network->dw1, 0, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    memset(network->db1, 0, HIDDEN_SIZE * sizeof(float));
    memset(network->dw2, 0, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    memset(network->db2, 0, OUTPUT_SIZE * sizeof(float));
}

void network_forward(const Network *network,
                     const float *input,
                     float *hidden,
                     float *logits,
                     float *probabilities) {
    int hidden_index = 0;
    int output_index = 0;

    for (hidden_index = 0; hidden_index < HIDDEN_SIZE; ++hidden_index) {
        float sum = network->b1[hidden_index];
        const float *weights = network->w1 + (size_t)hidden_index * INPUT_SIZE;

        sum += dot_product(weights, input, INPUT_SIZE);

        hidden[hidden_index] = relu(sum);
    }

    for (output_index = 0; output_index < OUTPUT_SIZE; ++output_index) {
        float sum = network->b2[output_index];
        const float *weights = network->w2 + (size_t)output_index * HIDDEN_SIZE;

        sum += dot_product(weights, hidden, HIDDEN_SIZE);

        logits[output_index] = sum;
    }

    softmax(logits, probabilities);
}

float network_accumulate_gradients(Network *network,
                                   const float *input,
                                   unsigned char label,
                                   const float *hidden,
                                   const float *probabilities) {
    float output_delta[OUTPUT_SIZE];
    float hidden_delta[HIDDEN_SIZE];
    float loss = 0.0f;
    int output_index = 0;
    int hidden_index = 0;

    for (output_index = 0; output_index < OUTPUT_SIZE; ++output_index) {
        output_delta[output_index] = probabilities[output_index] - (output_index == (int)label ? 1.0f : 0.0f);
    }

    loss = -logf(probabilities[label] + 1e-7f);

    for (output_index = 0; output_index < OUTPUT_SIZE; ++output_index) {
        const float delta = output_delta[output_index];
        float *weights_grad = network->dw2 + (size_t)output_index * HIDDEN_SIZE;

        network->db2[output_index] += delta;
        axpy_inplace(weights_grad, hidden, delta, HIDDEN_SIZE);
    }

    for (hidden_index = 0; hidden_index < HIDDEN_SIZE; ++hidden_index) {
        float gradient = 0.0f;

        for (output_index = 0; output_index < OUTPUT_SIZE; ++output_index) {
            gradient += output_delta[output_index] * network->w2[(size_t)output_index * HIDDEN_SIZE + hidden_index];
        }

        hidden_delta[hidden_index] = hidden[hidden_index] > 0.0f ? gradient : 0.0f;
        network->db1[hidden_index] += hidden_delta[hidden_index];

        axpy_inplace(network->dw1 + (size_t)hidden_index * INPUT_SIZE, input, hidden_delta[hidden_index], INPUT_SIZE);
    }

    return loss;
}

void network_apply_gradients(Network *network, float learning_rate, int batch_size) {
    float scale = learning_rate / (float)batch_size;

    axpy_inplace(network->w1, network->dw1, -scale, INPUT_SIZE * HIDDEN_SIZE);
    axpy_inplace(network->b1, network->db1, -scale, HIDDEN_SIZE);
    axpy_inplace(network->w2, network->dw2, -scale, HIDDEN_SIZE * OUTPUT_SIZE);
    axpy_inplace(network->b2, network->db2, -scale, OUTPUT_SIZE);
}

int network_predict(const Network *network,
                    const float *input,
                    float *workspace_hidden,
                    float *workspace_logits,
                    float *workspace_probabilities) {
    int best_index = 0;
    int i = 0;

    network_forward(network, input, workspace_hidden, workspace_logits, workspace_probabilities);

    for (i = 1; i < OUTPUT_SIZE; ++i) {
        if (workspace_probabilities[i] > workspace_probabilities[best_index]) {
            best_index = i;
        }
    }

    return best_index;
}

float network_accuracy(const Network *network, const float *images, const unsigned char *labels, int count) {
    float hidden[HIDDEN_SIZE];
    float logits[OUTPUT_SIZE];
    float probabilities[OUTPUT_SIZE];
    int correct = 0;
    int i = 0;

    for (i = 0; i < count; ++i) {
        int prediction = network_predict(network,
                                         images + (size_t)i * INPUT_SIZE,
                                         hidden,
                                         logits,
                                         probabilities);
        if (prediction == labels[i]) {
            ++correct;
        }
    }

    return (float)correct / (float)count;
}

void network_load_parameters(Network *network,
                             const float *w1,
                             const float *b1,
                             const float *w2,
                             const float *b2) {
    memcpy(network->w1, w1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    memcpy(network->b1, b1, HIDDEN_SIZE * sizeof(float));
    memcpy(network->w2, w2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    memcpy(network->b2, b2, OUTPUT_SIZE * sizeof(float));
}
