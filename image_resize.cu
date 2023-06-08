#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// CPU implementation with sequential execution
void resizeImageCPU(const uchar* input, uchar* output, int inputWidth, int inputHeight, int outputWidth, int outputHeight)
{
    for (int outputY = 0; outputY < outputHeight; ++outputY) {
        for (int outputX = 0; outputX < outputWidth; ++outputX) {
            int inputX = outputX * 2;
            int inputY = outputY * 2;

            uchar p00 = input[inputY * inputWidth + inputX];
            uchar p01 = input[inputY * inputWidth + inputX + 1];
            uchar p10 = input[(inputY + 1) * inputWidth + inputX];
            uchar p11 = input[(inputY + 1) * inputWidth + inputX + 1];

            output[outputY * outputWidth + outputX] = (p00 + p01 + p10 + p11) / 4;
        }
    }
}

__global__ void resizeImageCUDA(const uchar* input, uchar* output, int inputWidth, int inputHeight, int outputWidth, int outputHeight)
{
    int outputX = blockIdx.x * blockDim.x + threadIdx.x;
    int outputY = blockIdx.y * blockDim.y + threadIdx.y;

    if (outputX < outputWidth && outputY < outputHeight) {
        int inputX = outputX * 2;
        int inputY = outputY * 2;

        uchar p00 = input[inputY * inputWidth + inputX];
        uchar p01 = input[inputY * inputWidth + inputX + 1];
        uchar p10 = input[(inputY + 1) * inputWidth + inputX];
        uchar p11 = input[(inputY + 1) * inputWidth + inputX + 1];

        output[outputY * outputWidth + outputX] = (p00 + p01 + p10 + p11) / 4;
    }
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        printf("Usage: ./image_resize input_image output_image num_threads\n");
        return -1;
    }

    const char* inputImagePath = argv[1];
    const char* outputImagePath = argv[2];
    int numThreads = atoi(argv[3]);

    cv::Mat inputImage = cv::imread(inputImagePath, cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        printf("Could not read the input image: %s\n", inputImagePath);
        return -1;
    }

    int inputWidth = inputImage.cols;
    int inputHeight = inputImage.rows;

    // Kontrol: Boyutlar 2'ye tam bölünebilir mi?
    if (inputWidth % 2 != 0 || inputHeight % 2 != 0) {
        printf("Input image dimensions must be divisible by 2.\n");
        return -1;
    }

    int outputWidth = inputWidth / 2;
    int outputHeight = inputHeight / 2;

    cv::Mat outputImage(outputHeight, outputWidth, CV_8UC1);

    uchar* d_input;
    uchar* d_output;

    cudaMalloc((void**)&d_input, inputWidth * inputHeight * sizeof(uchar));
    cudaMalloc((void**)&d_output, outputWidth * outputHeight * sizeof(uchar));

    cudaMemcpy(d_input, inputImage.data, inputWidth * inputHeight * sizeof(uchar), cudaMemcpyHostToDevice);

    dim3 blockSize(numThreads, numThreads);
    dim3 gridSize((outputWidth + blockSize.x - 1) / blockSize.x, (outputHeight + blockSize.y - 1) / blockSize.y);

    clock_t start = clock();

    resizeImageCUDA<<<gridSize, blockSize>>>(d_input, d_output, inputWidth, inputHeight, outputWidth, outputHeight);

    cudaDeviceSynchronize();

    clock_t end = clock();

    cudaMemcpy(outputImage.data, d_output, outputWidth * outputHeight * sizeof(uchar), cudaMemcpyDeviceToHost);

    cv::imwrite(outputImagePath, outputImage);

    cudaFree(d_input);
    cudaFree(d_output);

    double cudaTime = (double)(end - start) / CLOCKS_PER_SEC;

    // Sequential CPU execution
    uchar* inputPtr = inputImage.data;
    uchar* outputPtr = outputImage.data;

    start = clock();

    resizeImageCPU(inputPtr, outputPtr, inputWidth, inputHeight, outputWidth, outputHeight);

    end = clock();

    double cpuTime = (double)(end - start) / CLOCKS_PER_SEC;

    // Speedup calculation
    double speedup = cpuTime / cudaTime;

    printf("Sequential CPU time: %f seconds\n", cpuTime);
    printf("CUDA time: %f seconds\n", cudaTime);
    printf("Speedup: %f\n", speedup);

    return 0;
}
