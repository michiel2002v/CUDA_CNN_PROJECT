#include <vector>
#include <memory>
#include <random>
#include <cuda_runtime.h>

#ifndef LAYER_H
#define LAYER_H
#endif

const static float threshold = 1.0E-01f;  // we accept current weights because there is 99% accuracy

class Layer {
public:
    int filter_size;             // size of the filer/kernel in 1D  (=M)
    int featuremaps;             // number of feature maps we want to apply (=N)
    int output_size;             // output size of the layer in 1D  (=O)

    float *output;              // output of the layer
    float *preact;              // result before applying activation function
    float *bias;                // bias weights
    float *weight;              // filter weights

    float *der_output;          // derivate/gradient of back propagation
    float *der_preact;
    float *der_weight;

    float *error;


    Layer(int filter_size, int featuremaps, int output_size);
    ~Layer();

    void setOutput(float *data);  // todo: we kunnen ipv telkens image per image naar gpu memory te kopiëren, allemaal in één keer doen
    void clear();
    void bp_clear();

};


// Utility CUDA kernel functions
__global__ void euclidianNorm(int n, float *x, float *result);
__device__ float step_function(float v);
__global__ void apply_step_function(float *input, float *output, const int N);
__global__ void squareElements(float* x, float* y, int n);
__global__ void makeError(float *err, float *output, unsigned int Y, const int N);
__global__ void apply_grad(float *output, float *grad, const int N);

// Forward propagation kernels
__global__ void fp_preact_c1(float input[28][28], float preact[6][24][24], float weight[6][5][5]);
__global__ void fp_bias_c1(float preact[6][24][24], float bias[6]);
__global__ void fp_preact_s1(float input[6][24][24], float preact[6][6][6], float weight[1][4][4]);
__global__ void fp_bias_s1(float preact[6][6][6], float bias[1]);
__global__ void fp_preact_f(float input[6][6][6], float preact[10], float weight[10][6][6][6]);
__global__ void fp_bias_f(float preact[10], float bias[10]);

// Back propagation kernels
__global__ void bp_weight_f(float d_weight[10][6][6][6], float d_preact[10], float p_output[6][6][6]);
__global__ void bp_bias_f(float bias[10], float d_preact[10]);
__global__ void bp_output_s1(float d_output[6][6][6], float n_weight[10][6][6][6], float nd_preact[10]);
__global__ void bp_preact_s1(float d_preact[6][6][6], float d_output[6][6][6], float preact[6][6][6]);
__global__ void bp_weight_s1(float d_weight[1][4][4], float d_preact[6][6][6], float p_output[6][24][24]);
__global__ void bp_bias_s1(float bias[1], float d_preact[6][6][6]);
__global__ void bp_output_c1(float d_output[6][24][24], float n_weight[1][4][4], float nd_preact[6][6][6]);
__global__ void bp_preact_c1(float d_preact[6][24][24], float d_output[6][24][24], float preact[6][24][24]);
__global__ void bp_weight_c1(float d_weight[6][5][5], float d_preact[6][24][24], float p_output[28][28]);
__global__ void bp_bias_c1(float bias[6], float d_preact[6][24][24]);
