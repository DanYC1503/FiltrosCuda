#pragma once

#include <vector>

// Genera una máscara de convolución para Motion Blur
std::vector<float> generateMotionBlurKernel(int size);

// Aplica convolución secuencial a una imagen RGB
void applyConvolutionSequentialRGB(
    const unsigned char* input,
    unsigned char* output,
    int width, int height,
    const std::vector<float>& kernel,
    int kernelSize
);

// Aplica convolución en paralelo usando CUDA para imagen RGB
void applyConvolutionParallelRGB(
    const unsigned char* h_inputImage,
    unsigned char* h_outputImageMean,
    unsigned char* h_outputImageBlur,
    int width, int height,
    int channels,
    const std::vector<float>& kernel,
    int kernelSize
);
// Para CPU
void mean_filter_cpu(
    const unsigned char* input, 
    unsigned char* output, 
    int width, 
    int height, 
    int channels, 
    int kernelSize,
    const std::vector<float>& kernel
);
void generateDoGImage(const float* blur1, const float* blur2, float* dog, int size);
double calculatePercentageError(const float* img1, const float* img2, int size);

void convolutionHost(
    const unsigned char* input,
    float* output,
    int width, int height,
    int channels,
    const float* mask,
    int maskSize
);
void printMask(const float* mask, int maskSize);
void createGaussianMask(float* mask, int maskSize, float sigma);