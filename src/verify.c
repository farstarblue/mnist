#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "config.h"
#include "mnist.h"
#include "model_params.h"
#include "network.h"

static void print_ascii_image(const float *image) {
    static const char levels[] = " .:-=+*#%@";
    int row = 0;
    int col = 0;

    for (row = 0; row < MNIST_ROWS; ++row) {
        for (col = 0; col < MNIST_COLS; ++col) {
            float pixel = image[row * MNIST_COLS + col];
            int index = (int)(pixel * 9.0f + 0.5f);
            if (index < 0) {
                index = 0;
            }
            if (index > 9) {
                index = 9;
            }
            putchar(levels[index]);
            putchar(levels[index]);
        }
        putchar('\n');
    }
}

int main(void) {
    MnistDataset test_set;
    Network network;
    float features[FEATURE_SIZE];
    float logits[OUTPUT_SIZE];
    float probabilities[OUTPUT_SIZE];
    int sample_index = 0;
    int prediction = 0;
    int seed = (int)time(NULL);
    int status = 1;
#if defined(__riscv)
    unsigned long long start = 0;
    unsigned long long end = 0;
#endif

    memset(&test_set, 0, sizeof(test_set));
    memset(&network, 0, sizeof(network));

#if defined(__riscv)
    __asm__ __volatile__("csrr %0, 0xc02" : "=r"(start) :: "memory");
#endif

    if (!MODEL_PARAMS_TRAINED) {
        fprintf(stderr, "尚未生成模型参数，请先执行 make train。\n");
        goto cleanup;
    }

    if (load_mnist_dataset(TEST_IMAGES_PATH, TEST_LABELS_PATH, TEST_SAMPLE_COUNT, &test_set) != 0) {
        goto cleanup;
    }

    if (network_init(&network) != 0) {
        goto cleanup;
    }
    network_load_parameters(&network, MODEL_CONV_W, MODEL_CONV_B, MODEL_FC_W, MODEL_FC_B);

    srand((unsigned int)seed);
    sample_index = rand() % (int)test_set.count;
    prediction = network_predict(&network,
                                 test_set.images + (size_t)sample_index * INPUT_SIZE,
                                 features,
                                 logits,
                                 probabilities);

    printf("随机样本编号: %d\n", sample_index);
    printf("真实标签: %u\n", test_set.labels[sample_index]);
    printf("字符画:\n\n");
    print_ascii_image(test_set.images + (size_t)sample_index * INPUT_SIZE);
    printf("\n预测结果: %d\n", prediction);
    printf("判定: %s\n", prediction == test_set.labels[sample_index] ? "CORRECT" : "WRONG");
    status = 0;

cleanup:
#if defined(__riscv)
    __asm__ __volatile__("csrr %0, 0xc02" : "=r"(end) :: "memory");
    printf("instret delta (main): %llu\n", end - start);
#endif
    network_free(&network);
    free_mnist_dataset(&test_set);
    return status;
}
