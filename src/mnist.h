#ifndef MNIST_H
#define MNIST_H

#include <stddef.h>

typedef struct {
    size_t count;
    float *images;
    unsigned char *labels;
} MnistDataset;

int load_mnist_dataset(const char *image_path,
                       const char *label_path,
                       size_t expected_count,
                       MnistDataset *dataset);
void free_mnist_dataset(MnistDataset *dataset);

#endif
