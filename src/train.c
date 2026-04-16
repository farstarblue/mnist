#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"
#include "mnist.h"
#include "model_io.h"
#include "network.h"

static void shuffle_indices(int *indices, int count) {
    int i = 0;

    for (i = count - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}

static int validate_training_config(void) {
    if (TRAIN_EPOCHS <= 0) {
        fprintf(stderr, "TRAIN_EPOCHS 必须是正整数。\n");
        return -1;
    }
    if (TRAIN_BATCH_SIZE <= 0) {
        fprintf(stderr, "TRAIN_BATCH_SIZE 必须是正整数。\n");
        return -1;
    }
    if (TRAIN_LEARNING_RATE <= 0.0f) {
        fprintf(stderr, "TRAIN_LEARNING_RATE 必须是正数。\n");
        return -1;
    }

    return 0;
}

int main(void) {
    MnistDataset train_set;
    MnistDataset test_set;
    Network network;
    int *indices = NULL;
    const int epochs = TRAIN_EPOCHS;
    const int batch_size = TRAIN_BATCH_SIZE;
    const int seed = TRAIN_RANDOM_SEED;
    const float learning_rate = TRAIN_LEARNING_RATE;
    int epoch = 0;
    int index = 0;
    int status = 1;

    memset(&train_set, 0, sizeof(train_set));
    memset(&test_set, 0, sizeof(test_set));
    memset(&network, 0, sizeof(network));

    if (validate_training_config() != 0) {
        goto cleanup;
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

    for (index = 0; index < (int)train_set.count; ++index) {
        indices[index] = index;
    }

    printf("开始训练: epochs=%d, batch=%d, lr=%.4f, seed=%d\n", epochs, batch_size, learning_rate, seed);

    for (epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;
        float hidden[HIDDEN_SIZE];
        float logits[OUTPUT_SIZE];
        float probabilities[OUTPUT_SIZE];
        int batch_start = 0;
        int batch_index = 0;
        const int total_batches = ((int)train_set.count + batch_size - 1) / batch_size;

        shuffle_indices(indices, (int)train_set.count);

        for (batch_start = 0; batch_start < (int)train_set.count; batch_start += batch_size) {
            float batch_loss = 0.0f;
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
                batch_loss += network_accumulate_gradients(&network, input, label, hidden, probabilities);
            }
            epoch_loss += batch_loss;
            network_apply_gradients(&network, learning_rate, batch_count);

            ++batch_index;
            printf("\repoch %d/%d  batch %d/%d  progress=%.2f%%  batch_loss=%.4f",
                   epoch + 1,
                   epochs,
                   batch_index,
                   total_batches,
                   (100.0f * (float)batch_index) / (float)total_batches,
                   batch_loss / (float)batch_count);
            fflush(stdout);
        }

        printf("\n");
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
