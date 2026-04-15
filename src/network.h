#ifndef NETWORK_H
#define NETWORK_H

typedef struct {
    float *w1;
    float *b1;
    float *w2;
    float *b2;
    float *dw1;
    float *db1;
    float *dw2;
    float *db2;
} Network;

int network_init(Network *network);
void network_free(Network *network);
void network_he_init(Network *network);
void network_zero_grad(Network *network);
void network_forward(const Network *network,
                     const float *input,
                     float *hidden,
                     float *logits,
                     float *probabilities);
float network_accumulate_gradients(Network *network,
                                   const float *input,
                                   unsigned char label,
                                   const float *hidden,
                                   const float *probabilities);
void network_apply_gradients(Network *network, float learning_rate, int batch_size);
int network_predict(const Network *network, const float *input, float *workspace_hidden, float *workspace_logits, float *workspace_probabilities);
float network_accuracy(const Network *network, const float *images, const unsigned char *labels, int count);
void network_load_parameters(Network *network,
                             const float *w1,
                             const float *b1,
                             const float *w2,
                             const float *b2);

#endif
