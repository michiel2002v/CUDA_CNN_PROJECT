#include "layer.h"
using namespace std;

Layer::Layer(int filter_size, int featuremaps, int output_size) {
    this->filter_size = filter_size;
    this->featuremaps = featuremaps;
    this->output_size = output_size;
    this->output = NULL;
    this->preact = NULL;
    this->bias = NULL;
    this->weight = NULL;
    this->error = NULL;

    vector<float> h_bias(featuremaps);
    vector<vector<float>> h_weight(featuremaps, vector<float>(filter_size));

    // define the bias and the weights as random values between -0.5 and 0.5
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(-0.5f, 0.5f);

    for (int i=0; i<featuremaps; i++) {
        h_bias[i] = dis(gen);
        for (int j=0; j<filter_size; j++) {
            h_weight[i][j] = dis(gen);
        }
    }

    // make space in GPU memory
    cudaMalloc(&output, sizeof(float) * output_size);
    cudaMalloc(&preact, sizeof(float) * output_size);
    cudaMalloc(&bias, sizeof(float) * featuremaps);
    cudaMalloc(&weight, sizeof(float) * filter_size * featuremaps);
    cudaMalloc(&der_output, sizeof(float) * output_size);
    cudaMalloc(&der_preact, sizeof(float) * output_size);
    cudaMalloc(&der_weight, sizeof(float) * filter_size * featuremaps);
    cudaMalloc(&error, sizeof(float));

    // copy weights to GPU memory
    cudaMemcpy(bias, h_bias.data(), sizeof(float) * featuremaps, cudaMemcpyHostToDevice);
    cudaMemcpy(weight, h_weight.data(), sizeof(float) * filter_size * featuremaps, cudaMemcpyHostToDevice);
    cudaMemset(error, 0, sizeof(float));
}
Layer::~Layer()
{
    cudaFree(output);
    cudaFree(preact);
    cudaFree(bias);
    cudaFree(weight);
    cudaFree(der_output);
    cudaFree(der_preact);
    cudaFree(der_weight);
    cudaFree(error);
}

// Send data one row from dataset to the GPU
void Layer::setOutput(float *data)
{
    cudaMemcpy(output, data, sizeof(float) * output_size, cudaMemcpyHostToDevice);
}

// Reset GPU memory between iterations
void Layer::clear()
{
    cudaMemset(output, 0x00, sizeof(float) * output_size);
    cudaMemset(preact, 0x00, sizeof(float) * output_size);
}

// clear results of back propagation
void Layer::bp_clear()
{
    cudaMemset(der_output, 0x00, sizeof(float) * output_size);
    cudaMemset(der_preact, 0x00, sizeof(float) * output_size);
    cudaMemset(der_weight, 0x00, sizeof(float) * filter_size * featuremaps);
    cudaMemset(error, 0, sizeof(float));
}



// this is a sigmoid activation function
__device__ float step_function(float v)
{
    return 1 / (1 + exp(-v));
}

__global__ void apply_step_function(float *input, float *output, const int N)
{
//    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
//    const int size = blockDim.x * gridDim.x;
//
//    for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
//        output[idx] = step_function(input[idx]);
//    }
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int idx_z = blockIdx.x;
    int index = (idx_x * blockDim.y * gridDim.x) + (idx_y * gridDim.x) + idx_z;
    output[index] = step_function(input[index]);

}

__global__ void euclidianNorm(int n, float *x, float *result) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        atomicAdd(result, x[idx] * x[idx]);
    }
    __syncthreads();
    if (idx == 0) {
        *result = sqrt(*result);
    }
}

__global__ void makeError(float *err, float *output, unsigned int Y, const int N)
{
//    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
//    const int size = blockDim.x * gridDim.x;
//
//    for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
//        err[idx] = ((Y == idx ? 1.0f : 0.0f) - output[idx]);
//    }

    int idx_x = threadIdx.x;
    err[idx_x] = ((Y == idx_x ? 1.0f : 0.0f) - output[idx_x]);
}

__global__ void apply_grad(float *output, float *grad, const int N)
{
    // Adds a fraction of the derived weight to the original weight

    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int idx_z = threadIdx.z;
    int indx_block = blockIdx.x;
    int index = (idx_z * blockDim.y * blockDim.x * gridDim.x) + (idx_y * blockDim.x * gridDim.x) + (idx_x * gridDim.x) + indx_block;
    output[index] += 1.0E-01f * grad[index];
}

__global__ void fp_preact_c1(float input[28][28], float preact[6][24][24], float weight[6][5][5])
{
    // For 1 Block of 24x24 threads, every thread calculates 6 output values
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;

    if (idx_x < 24 && idx_y < 24) {
        for (int idx_z=0; idx_z < 6; idx_z++) {
            float sum = 0.0f;
            for (int i = 0; i < 5; ++i) {
                for (int j = 0; j < 5; ++j) {
                    sum += input[idx_x + i][idx_y + j] * weight[idx_z][i][j];
                }
            }
            preact[idx_z][idx_x][idx_y] = sum;
        }
    }
}

__global__ void fp_bias_c1(float preact[6][24][24], float bias[6])
{
    // 6 Blocks of 24x24 threads, there is one thread for every output value
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int idx_z = blockIdx.x;
    preact[idx_z][idx_x][idx_y] += bias[idx_z];
}

__global__ void fp_preact_s1(float input[6][24][24], float preact[6][6][6], float weight[1][4][4])
{
    // For 1 Block of 24x24 threads, every thread calculates multiple output values
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;

    if (idx_x < 6 && idx_y < 6) {
        for (int idx_z = 0; idx_z < 6; idx_z++) {
            float sum = 0.0f;
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    sum += weight[0][i][j] * input[idx_z][idx_x * 4 + i][idx_y * 4 + j];
                }
            }
            preact[idx_z][idx_x][idx_y] = sum;
        }
    }
}

__global__ void fp_bias_s1(float preact[6][6][6], float bias[1])
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int idx_z = blockIdx.x;
    preact[idx_z][idx_x][idx_y] += bias[idx_z];
}

__global__ void fp_preact_f(float input[6][6][6], float preact[10], float weight[10][6][6][6])  // todo: denk eens na dommerikken!!!
{
    // For 1 Block of 6x6x6 threads, every thread calculates multiple output values
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int idx_z = threadIdx.z;

    if (idx_x < 6 && idx_y < 6 && idx_z < 6) {
        for (int i = 0; i < 10; i++) {
            float sum = weight[i][idx_x][idx_y][idx_z] * input[idx_x][idx_y][idx_z];
            atomicAdd(&preact[i], sum);
        }
    }

}

__global__ void fp_bias_f(float preact[10], float bias[10])
{
    int idx_x = threadIdx.x;
    preact[idx_x] += bias[idx_x];
}

__global__ void bp_weight_f(float d_weight[10][6][6][6], float d_preact[10], float p_output[6][6][6])
{
    // 10 Block of 6x6x6 threads, every thread calculates one weight value
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int idx_z = threadIdx.z;
    int idx_block = blockIdx.x;

    if (idx_x < 6 && idx_y < 6 && idx_z < 6) {
        d_weight[idx_block][idx_x][idx_y][idx_z] = d_preact[idx_block] * p_output[idx_x][idx_y][idx_z];

    }
}

__global__ void bp_bias_f(float bias[10], float d_preact[10])
{

    // 1 block of 10 threads, every thread calculates one bias by taking 10% of the predicted output
    int idx = threadIdx.x;
    bias[idx] += 1.0E-01f * d_preact[idx];
}

__global__ void bp_output_s1(float d_output[6][6][6], float n_weight[10][6][6][6], float nd_preact[10])
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int idx_z = threadIdx.z;
    int idx_block = blockIdx.x;

    atomicAdd(&d_output[idx_x][idx_y][idx_z], n_weight[idx_block][idx_x][idx_y][idx_z] * nd_preact[idx_block]);
}

__global__ void bp_preact_s1(float d_preact[6][6][6], float d_output[6][6][6], float preact[6][6][6])
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int idx_z = threadIdx.z;
    const float output = step_function(preact[idx_x][idx_y][idx_z]);
    // we nemen de afgeleide van de derivate output
    d_preact[idx_x][idx_y][idx_z] = d_output[idx_x][idx_y][idx_z] * output * (1 - output);
}

__global__ void bp_weight_s1(float d_weight[1][4][4], float d_preact[6][6][6], float p_output[6][24][24])
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;

    if (idx_x < 4 && idx_y < 4) {
        float sum = 0.0f;
        for (int i4 = 0; i4 < 6; ++i4) {
            for (int i5 = 0; i5 < 6; ++i5) {
                for (int i6 = 0; i6 < 6; ++i6) {
                    sum += d_preact[i4][i5][i6] * p_output[i4][i5 * 4 + idx_x][i6 * 4 + idx_y];
                }
            }
        }
        atomicAdd(&d_weight[0][idx_x][idx_y], sum);
    }
}

__global__ void bp_bias_s1(float bias[1], float d_preact[6][6][6])
{
    // 1 Block of 6x6x6 threads, every thread does one calculation
    const float d = pow(6.0f, 3.0f);

    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int idx_z = threadIdx.z;

    atomicAdd(&bias[0], 1.0E-01f * d_preact[idx_x][idx_y][idx_z] / d);
}

__global__ void bp_output_c1(float d_output[6][24][24], float n_weight[1][4][4], float nd_preact[6][6][6])
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int idx_z = threadIdx.z;

    for (int i1=0; i1 <4; i1++) {
        for (int i2=0; i2<4; i2++) {
            d_output[idx_x][idx_y*4 +i1][idx_z*4 + i2] += n_weight[0][i1][i2] * nd_preact[idx_x][idx_y][idx_z];
        }
    }
}

__global__ void bp_preact_c1(float d_preact[6][24][24], float d_output[6][24][24], float preact[6][24][24])
{
    int idx_x = blockIdx.x;
    int idx_y = threadIdx.x;
    int idx_z = threadIdx.y;
    const float output = step_function(preact[idx_x][idx_y][idx_z]);
    // we nemen de afgeleide van de derivate output
    d_preact[idx_x][idx_y][idx_z] = d_output[idx_x][idx_y][idx_z] * output * (1 - output);
}

__global__ void bp_weight_c1(float d_weight[6][5][5], float d_preact[6][24][24], float p_output[28][28])
{
    int idx_x = blockIdx.x;
    int idx_y = threadIdx.x;
    int idx_z = threadIdx.y;
    const float d = pow(24.0f, 2.0f);
    float sum = 0.0f;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            sum += d_preact[idx_x][idx_y + i][idx_z + j] * p_output[idx_y + i][idx_z + j];
        }
    }
    d_weight[idx_x][idx_y][idx_z] = sum;

}

__global__ void bp_bias_c1(float bias[6], float d_preact[6][24][24])
{
    const float d = pow(6.0f, 3.0f);

    int idx_x = blockIdx.x;
    int idx_y = threadIdx.x;
    int idx_z = threadIdx.y;

    atomicAdd(&bias[idx_x], 1.0E-01f * d_preact[idx_x][idx_y][idx_z] / d);
}
