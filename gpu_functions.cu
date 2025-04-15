#include "functions.h"
#include "book.h"  
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <algorithm>
#include <iostream>

__global__ void convolutionKernelRGB(
    unsigned char* d_input,
    unsigned char* d_output,
    int width, int height,
    const float* d_kernel,
    int kernelSize
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = kernelSize / 2;

    if (x >= width || y >= height) return;

    float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;

    for (int ky = -offset; ky <= offset; ++ky) {
        for (int kx = -offset; kx <= offset; ++kx) {
            int ix = x + kx;
            int iy = y + ky;

            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                int imageIdx = (iy * width + ix) * 3;
                int kernelIdx = (ky + offset) * kernelSize + (kx + offset);

                sumR += d_input[imageIdx + 0] * d_kernel[kernelIdx];
                sumG += d_input[imageIdx + 1] * d_kernel[kernelIdx];
                sumB += d_input[imageIdx + 2] * d_kernel[kernelIdx];
            }
        }
    }

    int outIdx = (y * width + x) * 3;
    d_output[outIdx + 0] = static_cast<unsigned char>(fminf(fmaxf(sumR, 0.0f), 255.0f));
    d_output[outIdx + 1] = static_cast<unsigned char>(fminf(fmaxf(sumG, 0.0f), 255.0f));
    d_output[outIdx + 2] = static_cast<unsigned char>(fminf(fmaxf(sumB, 0.0f), 255.0f));
}
void applyConvolutionParallelRGB(
    const unsigned char* h_inputImage,
    unsigned char* h_outputImage,
    int width, int height,
    const std::vector<float>& kernel,
    int kernelSize
) {
    // GPU INFO
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name << "\n";
    }

    int currentDevice;
    cudaGetDevice(&currentDevice);
    cudaDeviceProp currentProp;
    cudaGetDeviceProperties(&currentProp, currentDevice);

    std::cout << "Currently using GPU: " << currentProp.name << "\n";

    // === Logica Convolucion ===
    int imageSize = width * height * 3;
    int kernelBytes = kernelSize * kernelSize * sizeof(float);

    unsigned char* d_input;
    unsigned char* d_output;
    float* d_kernel;

    HANDLE_ERROR(cudaMalloc(&d_input, imageSize));
    HANDLE_ERROR(cudaMalloc(&d_output, imageSize));
    HANDLE_ERROR(cudaMalloc(&d_kernel, kernelBytes));

    HANDLE_ERROR(cudaMemcpy(d_input, h_inputImage, imageSize, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_kernel, kernel.data(), kernelBytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(32, 32); 
    dim3 numBlocks((width + 31) / 32, (height + 31) / 32); 


    convolutionKernelRGB << <numBlocks, threadsPerBlock >> > (
        d_input, d_output, width, height, d_kernel, kernelSize
        );

    HANDLE_ERROR(cudaDeviceSynchronize());

    HANDLE_ERROR(cudaMemcpy(h_outputImage, d_output, imageSize, cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}

