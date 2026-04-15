#include "mnist.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"

static uint32_t read_u32_be(FILE *file) {
    unsigned char bytes[4];

    if (fread(bytes, 1, sizeof(bytes), file) != sizeof(bytes)) {
        return 0;
    }

    return ((uint32_t)bytes[0] << 24U) |
           ((uint32_t)bytes[1] << 16U) |
           ((uint32_t)bytes[2] << 8U) |
           (uint32_t)bytes[3];
}

int load_mnist_dataset(const char *image_path,
                       const char *label_path,
                       size_t expected_count,
                       MnistDataset *dataset) {
    FILE *image_file = NULL;
    FILE *label_file = NULL;
    unsigned char *image_bytes = NULL;
    size_t image_bytes_count = 0;
    uint32_t image_magic = 0;
    uint32_t label_magic = 0;
    uint32_t image_count = 0;
    uint32_t label_count = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
    size_t i = 0;

    memset(dataset, 0, sizeof(*dataset));

    image_file = fopen(image_path, "rb");
    label_file = fopen(label_path, "rb");
    if (image_file == NULL || label_file == NULL) {
        fprintf(stderr, "无法打开 MNIST 文件。请先执行 make data。\n");
        goto fail;
    }

    image_magic = read_u32_be(image_file);
    image_count = read_u32_be(image_file);
    rows = read_u32_be(image_file);
    cols = read_u32_be(image_file);

    label_magic = read_u32_be(label_file);
    label_count = read_u32_be(label_file);

    if (image_magic != 2051U || label_magic != 2049U) {
        fprintf(stderr, "MNIST 文件头无效。\n");
        goto fail;
    }

    if (image_count != label_count || image_count != expected_count) {
        fprintf(stderr, "MNIST 样本数量不匹配。\n");
        goto fail;
    }

    if (rows != MNIST_ROWS || cols != MNIST_COLS) {
        fprintf(stderr, "MNIST 图像尺寸异常。\n");
        goto fail;
    }

    dataset->count = image_count;
    dataset->images = (float *)malloc(dataset->count * INPUT_SIZE * sizeof(float));
    dataset->labels = (unsigned char *)malloc(dataset->count * sizeof(unsigned char));
    image_bytes_count = dataset->count * INPUT_SIZE;
    image_bytes = (unsigned char *)malloc(image_bytes_count);

    if (dataset->images == NULL || dataset->labels == NULL || image_bytes == NULL) {
        fprintf(stderr, "MNIST 内存分配失败。\n");
        goto fail;
    }

    if (fread(dataset->labels, 1, dataset->count, label_file) != dataset->count) {
        fprintf(stderr, "读取 MNIST 标签失败。\n");
        goto fail;
    }

    if (fread(image_bytes, 1, image_bytes_count, image_file) != image_bytes_count) {
        fprintf(stderr, "读取 MNIST 图像失败。\n");
        goto fail;
    }

    for (i = 0; i < image_bytes_count; ++i) {
        dataset->images[i] = image_bytes[i] / 255.0f;
    }

    free(image_bytes);
    fclose(image_file);
    fclose(label_file);
    return 0;

fail:
    free(image_bytes);
    free_mnist_dataset(dataset);
    if (image_file != NULL) {
        fclose(image_file);
    }
    if (label_file != NULL) {
        fclose(label_file);
    }
    return -1;
}

void free_mnist_dataset(MnistDataset *dataset) {
    if (dataset == NULL) {
        return;
    }

    free(dataset->images);
    free(dataset->labels);
    dataset->images = NULL;
    dataset->labels = NULL;
    dataset->count = 0;
}
