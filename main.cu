#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include <cuda_runtime.h>
#include <iostream>
//#include <cublas_v2.h>
#include "layer.h"

using namespace std;


static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

// Define layers of CNN
static Layer inputLayer = Layer(0, 0, 28*28);
static Layer convolutionLayer = Layer(5*5, 6, 24*24*6);			// Convolution layer
static Layer poolingLayer = Layer(4*4, 1, 6*6*6);				// Subsampling layer
static Layer connectedLayer = Layer(6*6*6, 10, 10);        // Fully connected layer

static void learn();
static unsigned int classify(double data[28][28]);
static void test();
static void forward_pass(double data[28][28]);
static void back_pass();

static inline void loaddata()
{
    int result = mnist_load("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
                            &train_set, &train_cnt);
//    int result = mnist_load("numbers_mnist/train-images.idx3-ubyte", "numbers_mnist/train-labels.idx1-ubyte",
//                            &train_set, &train_cnt);

    if (result == 0) cout << "succes" << endl;
    else cout << "fail "  << result << endl;

    result = mnist_load("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte",
                        &test_set, &test_cnt);
//    result = mnist_load("numbers_mnist/t10k-images.idx3-ubyte", "numbers_mnist/t10k-labels.idx1-ubyte",
//                        &test_set, &test_cnt);

    if (result == 0) cout << "succes" << endl;
    else cout << "fail "  << result << endl;
}


static void forward_pass(double data[28][28]) {
    float input[28][28];

    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            input[i][j] = data[i][j];
        }
    }

    inputLayer.clear();
    convolutionLayer.clear();
    poolingLayer.clear();
    connectedLayer.clear();
    float dt_host = 1.0E-01f;

    inputLayer.setOutput((float *)input);

    // Convolutional Layer Forward Pass
    dim3 threadsPerBlock(24,24);
    dim3 nBlocks(1);
    fp_preact_c1<<<nBlocks, threadsPerBlock>>>((float (*)[28])inputLayer.output, (float (*)[24][24])convolutionLayer.preact, (float (*)[5][5])convolutionLayer.weight);
    threadsPerBlock = dim3(24,24);
    nBlocks = dim3(6);
    fp_bias_c1<<<nBlocks, threadsPerBlock>>>((float (*)[24][24])convolutionLayer.preact, convolutionLayer.bias);
    apply_step_function<<<nBlocks, threadsPerBlock>>>(convolutionLayer.preact, convolutionLayer.output, convolutionLayer.output_size);

    // Subsampling Layer Forward Pass
    threadsPerBlock = dim3(6,6);
    nBlocks = dim3(1);
    fp_preact_s1<<<nBlocks, threadsPerBlock>>>((float (*)[24][24])convolutionLayer.output, (float (*)[6][6])poolingLayer.preact, (float (*)[4][4])poolingLayer.weight);
    threadsPerBlock = dim3(6,6);
    nBlocks = dim3(6);
    fp_bias_s1<<<nBlocks, threadsPerBlock>>>((float (*)[6][6])poolingLayer.preact, poolingLayer.bias);
    apply_step_function<<<nBlocks, threadsPerBlock>>>(poolingLayer.preact, poolingLayer.output, poolingLayer.output_size);

    // Fully Connected Layer Forward Pass
    threadsPerBlock = dim3(6,6,6);
    nBlocks = dim3(1);
    fp_preact_f<<<nBlocks, threadsPerBlock>>>((float (*)[6][6])poolingLayer.output, connectedLayer.preact, (float (*)[6][6][6])connectedLayer.weight);
    threadsPerBlock = dim3(10);
    nBlocks = dim3(1);
    fp_bias_f<<<nBlocks, threadsPerBlock>>>(connectedLayer.preact, connectedLayer.bias);
    apply_step_function<<<nBlocks, threadsPerBlock>>>(connectedLayer.preact, connectedLayer.output, connectedLayer.output_size);

}

// Back propagation to update weights
static void back_pass()
{
    // Backpropagation in the Fully Connected Layer
    dim3 nBlocks(10);
    dim3 threadsPerBlock(6,6,6);
    bp_weight_f<<<nBlocks, threadsPerBlock>>>((float (*)[6][6][6])connectedLayer.der_weight, connectedLayer.der_preact, (float (*)[6][6])poolingLayer.output);
    nBlocks = dim3(1);
    threadsPerBlock = dim3(10);
    bp_bias_f<<<nBlocks, threadsPerBlock>>>(connectedLayer.bias, connectedLayer.der_preact);

    // Backpropagation in the Subsampling Layer
    nBlocks = dim3(10);
    threadsPerBlock = dim3(6,6,6);
    bp_output_s1<<<nBlocks, threadsPerBlock>>>((float (*)[6][6])poolingLayer.der_output, (float (*)[6][6][6])connectedLayer.weight, connectedLayer.der_preact);
    nBlocks = dim3(1);
    bp_preact_s1<<<nBlocks, threadsPerBlock>>>((float (*)[6][6])poolingLayer.der_preact, (float (*)[6][6])poolingLayer.der_output, (float (*)[6][6])poolingLayer.preact);
    nBlocks = dim3(1);
    threadsPerBlock = dim3(4,4);
    bp_weight_s1<<<nBlocks, threadsPerBlock>>>((float (*)[4][4])poolingLayer.der_weight, (float (*)[6][6])poolingLayer.der_preact, (float (*)[24][24])convolutionLayer.output);
    threadsPerBlock = dim3(6,6,6);
    bp_bias_s1<<<nBlocks, threadsPerBlock>>>(poolingLayer.bias, (float (*)[6][6])poolingLayer.der_preact);

    // Backpropagation in the Convolutional Layer
    bp_output_c1<<<nBlocks, threadsPerBlock>>>((float (*)[24][24])convolutionLayer.der_output, (float (*)[4][4])poolingLayer.weight, (float (*)[6][6])poolingLayer.der_preact);
    nBlocks = dim3(6);
    threadsPerBlock = dim3(24,24);
    bp_preact_c1<<<nBlocks, threadsPerBlock>>>((float (*)[24][24])convolutionLayer.der_preact, (float (*)[24][24])convolutionLayer.der_output, (float (*)[24][24])convolutionLayer.preact);
    bp_weight_c1<<<nBlocks, threadsPerBlock>>>((float (*)[5][5])convolutionLayer.der_weight, (float (*)[24][24])convolutionLayer.der_preact, (float (*)[28])inputLayer.output);
    bp_bias_c1<<<nBlocks, threadsPerBlock>>>(convolutionLayer.bias, (float (*)[24][24])convolutionLayer.der_preact);

    // Weight Update
    nBlocks = dim3(10);
    threadsPerBlock = dim3(6,6,6);
    apply_grad<<<nBlocks, threadsPerBlock>>>(connectedLayer.weight, connectedLayer.der_weight, connectedLayer.filter_size * connectedLayer.featuremaps);
    nBlocks = dim3(1);
    threadsPerBlock = dim3(1,4,4);
    apply_grad<<<nBlocks, threadsPerBlock>>>(poolingLayer.weight, poolingLayer.der_weight, poolingLayer.filter_size * poolingLayer.featuremaps);
    threadsPerBlock = dim3(6,5,5);
    apply_grad<<<nBlocks, threadsPerBlock>>>(convolutionLayer.weight, convolutionLayer.der_weight, convolutionLayer.filter_size * convolutionLayer.featuremaps);
}


static void learn() {
    float error;
    int iterations = 50;
    double chrono = 0.0;

    cout << "Learning..." << endl;

    while (iterations-- > 0) {
        error = 0.0f;
        cout << "trainingcount: " << train_cnt << endl;
        for (int i=0; i<train_cnt; i++) {

            float host_err;
            float device_err;

            forward_pass(train_set[i].data);

            connectedLayer.bp_clear();
            poolingLayer.bp_clear();
            connectedLayer.bp_clear();

            dim3 nBlocks = dim3(1);
            dim3 threadsPerBlock = dim3(10);
            makeError<<<nBlocks, threadsPerBlock>>>(connectedLayer.der_preact, connectedLayer.output, train_set[i].label, 10);
            euclidianNorm<<<nBlocks, threadsPerBlock>>>(10, connectedLayer.der_preact, connectedLayer.error);
            cudaMemcpy(&host_err, &connectedLayer.error, sizeof(float), cudaMemcpyDeviceToHost);

            error += host_err;

            back_pass();
        }

        error /= train_cnt;
        std::cout << "error: " << std::scientific << error << ", time_on_gpu: " << fixed << chrono << endl;
        if (error < threshold) {
            cout << "Training complete, error less than threshold\n" << endl;
            break;
        }
    }
    // print time nog eens ofzo
}

static unsigned int classify(double data[28][28]) {
    float res[10];
    forward_pass(data);
    unsigned int max = 0;
    cudaMemcpy(res, connectedLayer.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);

    for (int i = 1; i < 10; ++i) {
        if (res[max] < res[i]) {
            max = i;
        }
    }

    return max;
}

static void test()
{
    int error = 0;

    for (int i = 0; i < train_cnt; ++i) {
        if (classify(train_set[i].data) != train_set[i].label) {
            ++error;
        }
    }
    std::cout << error << "/" << train_cnt << std::endl;
    std::cout << "Error Rate: " << static_cast<double>(error) / static_cast<double>(train_cnt) * 100.0 << "%" << std::endl;
}

static void unfold_input(double input[28][28], double unfolded[24*24][5*5])
{
    int a = 0;
//    unfold_input;

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            int b = 0;
            for (int x = i; x < i + 2; ++x)
                for (int y = j; y < j+2; ++y)
                    unfolded[a][b++] = input[x][y];
            a++;
        }
}

int main() {
    // Get the number of CUDA devices
//    int deviceCount;
//    cudaGetDeviceCount(&deviceCount);
//
//    for (int i = 0; i < deviceCount; ++i) {
//        cudaDeviceProp deviceProp;
//        cudaGetDeviceProperties(&deviceProp, i);
//
//        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
//        std::cout << "  Maximum threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
//        std::cout << "  Maximum threads per multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
//        std::cout << "  Maximum threads per GPU: " << deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount << std::endl;
//    }

//    return 0;
    loaddata();
    learn();
    test();


    return 0;
}
