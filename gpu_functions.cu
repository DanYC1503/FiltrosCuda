#include "functions.h"
#include "book.h"  
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <cmath>
#include "stb_image.h"
#include "stb_image_write.h"

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
__global__ void mean_filter_gpu(
    const unsigned char* input,
    unsigned char* output,
    int width, int height,
    int channels,
    int kernelSize
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = kernelSize / 2;

    if (x >= width || y >= height) return;

    // Usar la suma de los píxeles circundantes
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;

        // Sumar los valores de los píxeles en la región del kernel
        for (int dy = -offset; dy <= offset; dy++) {
            for (int dx = -offset; dx <= offset; dx++) {
                int nx = x + dx;
                int ny = y + dy;

                // Asegurarse de que esté dentro de los límites
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int imgIdx = (ny * width + nx) * channels + c;
                    sum += input[imgIdx];
                }
            }
        }

        // Calcular la media dividiendo entre el área del kernel (kernelSize * kernelSize)
        int outIdx = (y * width + x) * channels + c;
        output[outIdx] = (unsigned char)(fminf(fmaxf(sum / (kernelSize * kernelSize), 0.0f), 255.0f));
    }
}


void applyConvolutionParallelRGB(
    const unsigned char* h_inputImage,
    unsigned char* h_outputImageMean,
    unsigned char* h_outputImageBlur,
    int width, int height,
    int channels,
    const std::vector<float>& kernel,
    int kernelSize
) {
    int imageSize = width * height * channels;
    int kernelBytes = kernelSize * kernelSize * sizeof(float);
    size_t img_bytes = width * height * channels * sizeof(unsigned char);

    unsigned char* d_input;
    unsigned char* d_outputMean;  // Para el resultado del filtro de media
    unsigned char* d_outputBlur;  // Para el resultado del filtro de Motion Blur
    float* d_kernel;

    // Asignar memoria una vez
    HANDLE_ERROR(cudaMalloc(&d_input, imageSize));
    HANDLE_ERROR(cudaMalloc(&d_outputMean, imageSize));  // Para el filtro de media
    HANDLE_ERROR(cudaMalloc(&d_outputBlur, imageSize));  // Para el filtro de Motion Blur
    HANDLE_ERROR(cudaMalloc(&d_kernel, kernelBytes));

    // Copiar los datos de entrada a la GPU
    HANDLE_ERROR(cudaMemcpy(d_input, h_inputImage, imageSize, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_kernel, kernel.data(), kernelBytes, cudaMemcpyHostToDevice));

    // Configuración del kernel CUDA para 1024 hilos
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((width + 31) / 32, (height + 31) / 32);

    // Medición de tiempo para la convolución (Motion Blur)
    auto startConv = std::chrono::high_resolution_clock::now();

    // Ejecución del kernel de convolución (Motion Blur)
    convolutionKernelRGB << <numBlocks, threadsPerBlock >> > (
        d_input, d_outputBlur, width, height, d_kernel, kernelSize
        );
    HANDLE_ERROR(cudaDeviceSynchronize());  // Esperar a que termine la convolución

    auto endConv = std::chrono::high_resolution_clock::now();
    auto durationConv = std::chrono::duration_cast<std::chrono::milliseconds>(endConv - startConv).count();
    std::cout << "Filtro Motion Blur (GPU) tardó: " << durationConv << " ms\n";

    // Copiar el resultado de Motion Blur a la CPU inmediatamente
    HANDLE_ERROR(cudaMemcpy(h_outputImageBlur, d_outputBlur, img_bytes, cudaMemcpyDeviceToHost));

    // Guardar la imagen después del filtro de Motion Blur
    stbi_write_png("output_motion_blur.png", width, height, 3, h_outputImageBlur, width * 3);
    std::cout << "Imagen guardada después de Motion Blur\n";

    // Medición de tiempo para el filtro de media
    auto startMean = std::chrono::high_resolution_clock::now();

    // Ejecución del kernel del filtro de media (mismo buffer de salida para media)
    mean_filter_gpu << <numBlocks, threadsPerBlock >> > (
        d_input, d_outputMean, width, height, channels, kernelSize
        );
    HANDLE_ERROR(cudaDeviceSynchronize()); 

    auto endMean = std::chrono::high_resolution_clock::now();
    auto durationMean = std::chrono::duration_cast<std::chrono::milliseconds>(endMean - startMean).count();
    std::cout << "Filtro Media (GPU) tardó: " << durationMean << " ms\n";

    // Copiar el resultado del filtro de media a la CPU
    HANDLE_ERROR(cudaMemcpy(h_outputImageMean, d_outputMean, img_bytes, cudaMemcpyDeviceToHost));

    // Guardar la imagen después del filtro de media
    stbi_write_png("output_mean_filter.png", width, height, 3, h_outputImageMean, width * 3);
    std::cout << "Imagen guardada después de Mean Filter\n";

    // Depuración: comprobar los primeros píxeles de ambos resultados
    unsigned char* h_tempMean = new unsigned char[img_bytes];
    unsigned char* h_tempBlur = new unsigned char[img_bytes];

    HANDLE_ERROR(cudaMemcpy(h_tempMean, d_outputMean, img_bytes, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_tempBlur, d_outputBlur, img_bytes, cudaMemcpyDeviceToHost));

    std::cout << "Primer píxel en la salida de media: "
        << (int)h_tempMean[0] << ", "
        << (int)h_tempMean[1] << ", "
        << (int)h_tempMean[2] << "\n";

    std::cout << "Primer píxel en la salida de blur: "
        << (int)h_tempBlur[0] << ", "
        << (int)h_tempBlur[1] << ", "
        << (int)h_tempBlur[2] << "\n";

    delete[] h_tempMean;
    delete[] h_tempBlur;

    // Liberar memoria de la GPU
    cudaFree(d_input);
    cudaFree(d_outputMean);
    cudaFree(d_outputBlur);
    cudaFree(d_kernel);
}

