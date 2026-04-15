#include "model_io.h"

#include <stdio.h>

#include "config.h"

static void write_float_array(FILE *file, const char *name, const float *values, int count) {
    int i = 0;

    fprintf(file, "static const float %s[%d] = {\n", name, count);
    for (i = 0; i < count; ++i) {
        fprintf(file, "    %.9ef%s", values[i], (i + 1 == count) ? "" : ",");
        if ((i + 1) % 6 == 0 || i + 1 == count) {
            fputc('\n', file);
        } else {
            fputc(' ', file);
        }
    }
    fprintf(file, "};\n\n");
}

int save_model_header(const char *output_path, const Network *network) {
    FILE *file = fopen(output_path, "w");

    if (file == NULL) {
        fprintf(stderr, "无法写入模型头文件: %s\n", output_path);
        return -1;
    }

    fprintf(file, "#ifndef MODEL_PARAMS_H\n");
    fprintf(file, "#define MODEL_PARAMS_H\n\n");
    fprintf(file, "#include \"config.h\"\n\n");
    fprintf(file, "#define MODEL_PARAMS_TRAINED 1\n\n");

    write_float_array(file, "MODEL_W1", network->w1, INPUT_SIZE * HIDDEN_SIZE);
    write_float_array(file, "MODEL_B1", network->b1, HIDDEN_SIZE);
    write_float_array(file, "MODEL_W2", network->w2, HIDDEN_SIZE * OUTPUT_SIZE);
    write_float_array(file, "MODEL_B2", network->b2, OUTPUT_SIZE);

    fprintf(file, "#endif\n");
    fclose(file);
    return 0;
}
