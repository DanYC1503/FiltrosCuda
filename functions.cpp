#include "functions.h"
#include <algorithm> 
#include <cstring>
#include <iostream>

std::vector<float> generateMotionBlurKernel(int size) {
    std::vector<float> kernel(size * size, 0.0f);
    for (int i = 0; i < size; ++i) {
        kernel[i * size + i] = 1.0f / size;
    }
    return kernel;
}

void applyConvolutionSequentialRGB(
    const unsigned char* input,
    unsigned char* output,
    int width, int height,
    const std::vector<float>& kernel,
    int kernelSize
) {
    int offset = kernelSize / 2;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;

            for (int ky = -offset; ky <= offset; ++ky) {
                for (int kx = -offset; kx <= offset; ++kx) {
                    int ix = x + kx;
                    int iy = y + ky;

                    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                        int imageIndex = (iy * width + ix) * 3;
                        int kernelIndex = (ky + offset) * kernelSize + (kx + offset);

                        sumR += input[imageIndex + 0] * kernel[kernelIndex];
                        sumG += input[imageIndex + 1] * kernel[kernelIndex];
                        sumB += input[imageIndex + 2] * kernel[kernelIndex];
                    }
                }
            }

            int outIndex = (y * width + x) * 3;
            output[outIndex + 0] = static_cast<unsigned char>(std::min(std::max(sumR, 0.0f), 255.0f));
            output[outIndex + 1] = static_cast<unsigned char>(std::min(std::max(sumG, 0.0f), 255.0f));
            output[outIndex + 2] = static_cast<unsigned char>(std::min(std::max(sumB, 0.0f), 255.0f));

        }
    }
}

