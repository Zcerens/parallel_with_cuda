#include <iostream>
#include <cuda.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

__global__ void resizeImage(const uchar* input, uchar* output, int inputWidth, int inputHeight, int outputWidth, int outputHeight)
{
    int outputX = blockIdx.x * blockDim.x + threadIdx.x;
    int outputY = blockIdx.y * blockDim.y + threadIdx.y;

    if (outputX < outputWidth && outputY < outputHeight) {
        int inputX = outputX * 2;
        int inputY = outputY * 2;

        output[outputY * outputWidth + outputX] = (input[inputY * inputWidth + inputX] +
                                                   input[inputY * inputWidth + inputX + 1] +
                                                   input[(inputY + 1) * inputWidth + inputX] +
                                                   input[(inputY + 1) * inputWidth + inputX + 1]) / 4;
    }
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        cout << "Usage: ./image_resize input_image output_image num_threads" << endl;
        return -1;
    }

    string inputImagePath = argv[1];
    string outputImagePath = argv[2];
    int numThreads = atoi(argv[3]);

    Mat inputImage = imread(inputImagePath, IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        cout << "Could not read the input image: " << inputImagePath << endl;
        return -1;
    }

    int inputWidth = inputImage.cols;
    int inputHeight = inputImage.rows;
    int outputWidth = inputWidth / 2;
    int outputHeight = inputHeight / 2;

    Mat outputImage(outputHeight, outputWidth, CV_8UC1);

    uchar* d_input;
    uchar* d_output;

    cudaMalloc((void**)&d_input, inputWidth * inputHeight * sizeof(uchar));
    cudaMalloc((void**)&d_output, outputWidth * outputHeight * sizeof(uchar));

    cudaMemcpy(d_input, inputImage.data, inputWidth * inputHeight * sizeof(uchar), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((outputWidth + blockSize.x - 1) / blockSize.x, (outputHeight + blockSize.y - 1) / blockSize.y);

    clock_t start = clock();

    resizeImage<<<gridSize, blockSize>>>(d_input, d_output, inputWidth, inputHeight, outputWidth, outputHeight);

    clock_t end = clock();

    cudaMemcpy(outputImage.data, d_output, outputWidth * outputHeight * sizeof(uchar), cudaMemcpyDeviceToHost);

    imwrite(outputImagePath, outputImage);

    cudaFree(d_input);
    cudaFree(d_output);

    double time = (double)(end - start) / CLOCKS_PER_SEC;
    cout << "Image resized in " << time << " seconds." << endl;

    return 0;
}
