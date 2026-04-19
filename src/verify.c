#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "config.h"
#include "mnist.h"
#include "model_params.h"
#include "network.h"

static void print_usage(const char *program_name) {
    printf("用法: %s [--sample INDEX] [--repeat COUNT] [--num INDEX] [--random]\n", program_name);
    printf("  --sample INDEX  固定验证测试集中的第 INDEX 张图片\n");
    printf("  --repeat COUNT  对同一张图片重复推理 COUNT 次，用于基准对比\n");
    printf("  --num INDEX     验证测试集从 0 到 INDEX 的所有图片，并输出准确率\n");
    printf("  --random        使用随机样本（默认行为）\n");
    printf("  --help          显示帮助信息\n");
}

static int parse_int_arg(const char *text, int min_value, int *out_value) {
    char *end = NULL;
    long value = strtol(text, &end, 10);

    if (text[0] == '\0' || end == text || *end != '\0') {
        return -1;
    }
    if (value < (long)min_value || value > 2147483647L) {
        return -1;
    }

    *out_value = (int)value;
    return 0;
}

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

int main(int argc, char **argv) {
    MnistDataset test_set;
    Network network;
    float features[FEATURE_SIZE];
    float logits[OUTPUT_SIZE];
    float probabilities[OUTPUT_SIZE];
    int sample_index = 0;
    int prediction = 0;
    int seed = (int)time(NULL);
    int status = 1;
    int repeat_count = 1;
    int use_random_sample = 1;
    int use_range_eval = 0;
    int sample_was_set = 0;
    int max_sample_index = 0;
    int arg_index = 0;
    int evaluated_count = 0;
    float accuracy = 0.0f;
    const float *input = NULL;
#if defined(__riscv)
    unsigned long long start = 0;
    unsigned long long end = 0;
#endif

    memset(&test_set, 0, sizeof(test_set));
    memset(&network, 0, sizeof(network));

    for (arg_index = 1; arg_index < argc; ++arg_index) {
        if (strcmp(argv[arg_index], "--sample") == 0) {
            if (arg_index + 1 >= argc || parse_int_arg(argv[arg_index + 1], 0, &sample_index) != 0) {
                fprintf(stderr, "--sample 需要一个非负整数参数。\n");
                print_usage(argv[0]);
                goto cleanup;
            }
            use_random_sample = 0;
            sample_was_set = 1;
            ++arg_index;
        } else if (strcmp(argv[arg_index], "--repeat") == 0) {
            if (arg_index + 1 >= argc || parse_int_arg(argv[arg_index + 1], 1, &repeat_count) != 0) {
                fprintf(stderr, "--repeat 需要一个大于等于 1 的整数参数。\n");
                print_usage(argv[0]);
                goto cleanup;
            }
            ++arg_index;
        } else if (strcmp(argv[arg_index], "--num") == 0) {
            if (arg_index + 1 >= argc || parse_int_arg(argv[arg_index + 1], 0, &max_sample_index) != 0) {
                fprintf(stderr, "--num 需要一个非负整数参数。\n");
                print_usage(argv[0]);
                goto cleanup;
            }
            use_range_eval = 1;
            use_random_sample = 0;
            ++arg_index;
        } else if (strcmp(argv[arg_index], "--random") == 0) {
            use_random_sample = 1;
        } else if (strcmp(argv[arg_index], "--help") == 0) {
            print_usage(argv[0]);
            status = 0;
            goto cleanup;
        } else {
            fprintf(stderr, "未知参数: %s\n", argv[arg_index]);
            print_usage(argv[0]);
            goto cleanup;
        }
    }

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

    if (use_range_eval && sample_was_set) {
        fprintf(stderr, "--num 不能与 --sample 同时使用。\n");
        goto cleanup;
    }
    if (use_range_eval && repeat_count != 1) {
        fprintf(stderr, "--num 模式下不支持 --repeat，请直接统计 0..N 的准确率。\n");
        goto cleanup;
    }

    if (use_range_eval) {
        if ((size_t)max_sample_index >= test_set.count) {
            fprintf(stderr, "样本下标超出范围: %d，测试集大小为 %lu。\n",
                    max_sample_index,
                    (unsigned long)test_set.count);
            goto cleanup;
        }
        evaluated_count = max_sample_index + 1;
    } else if (use_random_sample) {
        srand((unsigned int)seed);
        sample_index = rand() % (int)test_set.count;
    } else if ((size_t)sample_index >= test_set.count) {
        fprintf(stderr, "样本下标超出范围: %d，测试集大小为 %lu。\n",
                sample_index,
                (unsigned long)test_set.count);
        goto cleanup;
    }

#if defined(__riscv)
    __asm__ __volatile__("csrr %0, 0xc02" : "=r"(start) :: "memory");
#endif
    if (use_range_eval) {
        accuracy = network_accuracy(&network, test_set.images, test_set.labels, evaluated_count);
    } else {
        input = test_set.images + (size_t)sample_index * INPUT_SIZE;
        for (arg_index = 0; arg_index < repeat_count; ++arg_index) {
            prediction = network_predict(&network, input, features, logits, probabilities);
        }
    }
#if defined(__riscv)
    __asm__ __volatile__("csrr %0, 0xc02" : "=r"(end) :: "memory");
#endif

    if (use_range_eval) {
        printf("样本模式: range\n");
        printf("验证范围: [0, %d]\n", max_sample_index);
        printf("验证样本数: %d\n", evaluated_count);
        printf("验证准确率: %.2f%%\n", accuracy * 100.0f);
    } else {
        printf("样本模式: %s\n", use_random_sample ? "random" : "fixed");
        printf("样本编号: %d\n", sample_index);
        printf("重复次数: %d\n", repeat_count);
        printf("真实标签: %u\n", test_set.labels[sample_index]);
        printf("字符画:\n\n");
        print_ascii_image(input);
        printf("\n预测结果: %d\n", prediction);
        printf("判定: %s\n", prediction == test_set.labels[sample_index] ? "CORRECT" : "WRONG");
    }
    status = 0;

cleanup:
#if defined(__riscv)
    if (status == 0) {
        unsigned long long delta = end - start;
        if (use_range_eval) {
            printf("instret total (accuracy %d samples): %llu\n", evaluated_count, delta);
            printf("instret avg   (per sample): %llu\n",
                   evaluated_count > 0 ? delta / (unsigned long long)evaluated_count : 0ULL);
        } else if (repeat_count > 1) {
            printf("instret total (predict x%d): %llu\n", repeat_count, delta);
            printf("instret avg   (predict): %llu\n", delta / (unsigned long long)repeat_count);
        } else {
            printf("instret delta (predict): %llu\n", delta);
        }
    }
#endif
    network_free(&network);
    free_mnist_dataset(&test_set);
    return status;
}
