#pragma once

#include <vector>

// Genera una m�scara de convoluci�n para Motion Blur
std::vector<float> generateMotionBlurKernel(int size);

// Aplica convoluci�n secuencial a una imagen RGB
void applyConvolutionSequentialRGB(
    const unsigned char* input,
    unsigned char* output,
    int width, int height,
    const std::vector<float>& kernel,
    int kernelSize
);

// Aplica convoluci�n en paralelo usando CUDA para imagen RGB
void applyConvolutionParallelRGB(
    const unsigned char* h_inputImage,
    unsigned char* h_outputImage,
    int width, int height,
    const std::vector<float>& kernel,
    int kernelSize
);
