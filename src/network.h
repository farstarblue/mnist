#ifndef NETWORK_H
#define NETWORK_H

typedef struct {
    float *conv_w;
    float *conv_b;
    float *fc_w;
    float *fc_b;
    float *dconv_w;
    float *dconv_b;
    float *dfc_w;
    float *dfc_b;
} Network;

int network_init(Network *network);
void network_free(Network *network);
void network_he_init(Network *network);
void network_zero_grad(Network *network);
void network_forward(const Network *network,
                     const float *input,
                     float *features,
                     float *logits,
                     float *probabilities);
float network_accumulate_gradients(Network *network,
                                   const float *input,
                                   unsigned char label,
                                   const float *features,
                                   const float *probabilities);
void network_apply_gradients(Network *network, float learning_rate, int batch_size);
int network_predict(const Network *network, const float *input, float *workspace_features, float *workspace_logits, float *workspace_probabilities);
float network_accuracy(const Network *network, const float *images, const unsigned char *labels, int count);
void network_load_parameters(Network *network,
                             const float *conv_w,
                             const float *conv_b,
                             const float *fc_w,
                             const float *fc_b);

#endif
