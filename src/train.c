#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "config.h"
#include "mnist.h"
#include "model_io.h"
#include "network.h"

#define TRAIN_IMAGES_PATH "data/train-images-idx3-ubyte"
#define TRAIN_LABELS_PATH "data/train-labels-idx1-ubyte"
#define TEST_IMAGES_PATH "data/t10k-images-idx3-ubyte"
#define TEST_LABELS_PATH "data/t10k-labels-idx1-ubyte"
#define MODEL_HEADER_PATH "src/model_params.h"

static void shuffle_indices(int *indices, int count) {
    int i = 0;

    for (i = count - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}

static int parse_int_arg(const char *name, const char *value, int *target) {
    char *end = NULL;
    long parsed = strtol(value, &end, 10);

    if (end == value || *end != '\0' || parsed <= 0) {
        fprintf(stderr, "%s 必须是正整数。\n", name);
        return -1;
    }

    *target = (int)parsed;
    return 0;
}

static int parse_float_arg(const char *name, const char *value, float *target) {
    char *end = NULL;
    float parsed = strtof(value, &end);

    if (end == value || *end != '\0' || parsed <= 0.0f) {
        fprintf(stderr, "%s 必须是正数。\n", name);
        return -1;
    }

    *target = parsed;
    return 0;
}

static void print_usage(const char *program) {
    printf("用法: %s [--epochs N] [--batch-size N] [--learning-rate X] [--seed N]\n", program);
}

int main(int argc, char **argv) {
    MnistDataset train_set;
    MnistDataset test_set;
    Network network;
    int *indices = NULL;
    int epochs = DEFAULT_EPOCHS;
    int batch_size = DEFAULT_BATCH_SIZE;
    int seed = (int)time(NULL);
    float learning_rate = DEFAULT_LEARNING_RATE;
    int epoch = 0;
    int argi = 0;
    int status = 1;

    memset(&train_set, 0, sizeof(train_set));
    memset(&test_set, 0, sizeof(test_set));
    memset(&network, 0, sizeof(network));

    for (argi = 1; argi < argc; ++argi) {
        if (strcmp(argv[argi], "--epochs") == 0 && argi + 1 < argc) {
            if (parse_int_arg("epochs", argv[++argi], &epochs) != 0) {
                goto cleanup;
            }
        } else if (strcmp(argv[argi], "--batch-size") == 0 && argi + 1 < argc) {
            if (parse_int_arg("batch-size", argv[++argi], &batch_size) != 0) {
                goto cleanup;
            }
        } else if (strcmp(argv[argi], "--learning-rate") == 0 && argi + 1 < argc) {
            if (parse_float_arg("learning-rate", argv[++argi], &learning_rate) != 0) {
                goto cleanup;
            }
        } else if (strcmp(argv[argi], "--seed") == 0 && argi + 1 < argc) {
            if (parse_int_arg("seed", argv[++argi], &seed) != 0) {
                goto cleanup;
            }
        } else if (strcmp(argv[argi], "--help") == 0) {
            print_usage(argv[0]);
            status = 0;
            goto cleanup;
        } else {
            print_usage(argv[0]);
            goto cleanup;
        }
    }

    srand((unsigned int)seed);

    if (load_mnist_dataset(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, TRAIN_SAMPLE_COUNT, &train_set) != 0) {
        goto cleanup;
    }

    if (load_mnist_dataset(TEST_IMAGES_PATH, TEST_LABELS_PATH, TEST_SAMPLE_COUNT, &test_set) != 0) {
        goto cleanup;
    }

    if (network_init(&network) != 0) {
        goto cleanup;
    }
    network_he_init(&network);

    indices = (int *)malloc(train_set.count * sizeof(int));
    if (indices == NULL) {
        fprintf(stderr, "索引内存分配失败。\n");
        goto cleanup;
    }

    for (argi = 0; argi < (int)train_set.count; ++argi) {
        indices[argi] = argi;
    }

    printf("开始训练: epochs=%d, batch=%d, lr=%.4f, seed=%d\n", epochs, batch_size, learning_rate, seed);

    for (epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;
        float hidden[HIDDEN_SIZE];
        float logits[OUTPUT_SIZE];
        float probabilities[OUTPUT_SIZE];
        int batch_start = 0;

        shuffle_indices(indices, (int)train_set.count);

        for (batch_start = 0; batch_start < (int)train_set.count; batch_start += batch_size) {
            int batch_count = batch_size;
            int batch_offset = 0;

            if (batch_start + batch_count > (int)train_set.count) {
                batch_count = (int)train_set.count - batch_start;
            }

            network_zero_grad(&network);
            for (batch_offset = 0; batch_offset < batch_count; ++batch_offset) {
                int sample_index = indices[batch_start + batch_offset];
                const float *input = train_set.images + (size_t)sample_index * INPUT_SIZE;
                unsigned char label = train_set.labels[sample_index];

                network_forward(&network, input, hidden, logits, probabilities);
                epoch_loss += network_accumulate_gradients(&network, input, label, hidden, probabilities);
            }
            network_apply_gradients(&network, learning_rate, batch_count);
        }

        printf("epoch %d/%d  loss=%.4f  test_acc=%.2f%%\n",
               epoch + 1,
               epochs,
               epoch_loss / (float)train_set.count,
               network_accuracy(&network, test_set.images, test_set.labels, (int)test_set.count) * 100.0f);
    }

    if (save_model_header(MODEL_HEADER_PATH, &network) != 0) {
        goto cleanup;
    }

    printf("模型参数已写入 %s\n", MODEL_HEADER_PATH);
    status = 0;

cleanup:
    free(indices);
    network_free(&network);
    free_mnist_dataset(&train_set);
    free_mnist_dataset(&test_set);
    return status;
}
