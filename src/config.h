#ifndef CONFIG_H
#define CONFIG_H

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10

#define MNIST_ROWS 28
#define MNIST_COLS 28
#define TRAIN_SAMPLE_COUNT 60000
#define TEST_SAMPLE_COUNT 10000

#define TRAIN_IMAGES_PATH "data/train-images-idx3-ubyte"
#define TRAIN_LABELS_PATH "data/train-labels-idx1-ubyte"
#define TEST_IMAGES_PATH "data/t10k-images-idx3-ubyte"
#define TEST_LABELS_PATH "data/t10k-labels-idx1-ubyte"
#define MODEL_HEADER_PATH "src/model_params.h"

#define TRAIN_EPOCHS 3
#define TRAIN_BATCH_SIZE 64
#define TRAIN_LEARNING_RATE 0.01f
#define TRAIN_RANDOM_SEED 42

#endif
