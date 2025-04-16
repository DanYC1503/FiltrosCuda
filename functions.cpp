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
void mean_filter_cpu(
    const unsigned char* input,
    unsigned char* output,
    int width, int height,
    int channels,
    int kernelSize,
    const std::vector<float>& kernel
) {
    int offset = kernelSize / 2;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                float sum = 0.0f;

                for (int dy = -offset; dy <= offset; dy++) {
                    for (int dx = -offset; dx <= offset; dx++) {
                        int nx = x + dx;
                        int ny = y + dy;

                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            int imgIdx = (ny * width + nx) * channels + c;
                            int kernelIdx = (dy + offset) * kernelSize + (dx + offset);
                            sum += input[imgIdx] * kernel[kernelIdx];
                        }
                    }
                }

                int outIdx = (y * width + x) * channels + c;
                output[outIdx] = static_cast<unsigned char>(std::min(std::max(sum, 0.0f), 255.0f));
            }
        }
    }
}

// Crea una máscara Gaussiana de tamaño maskSize x maskSize con desviación sigma
void createGaussianMask(float* mask, int maskSize, float sigma) {
    int offset = maskSize / 2;
    float sum = 0.0f;
    for (int y = -offset; y <= offset; y++) {
        for (int x = -offset; x <= offset; x++) {
            float value = expf(-(x * x + y * y) / (2.0f * sigma * sigma));
            mask[(y + offset) * maskSize + (x + offset)] = value;
            sum += value;
        }
    }
    for (int i = 0; i < maskSize * maskSize; i++) {
        mask[i] /= sum;
    }
}

// Imprime la máscara Gaussiana (para depuración o visualización)
void printMask(const float* mask, int maskSize) {
    for (int i = 0; i < maskSize; i++) {
        for (int j = 0; j < maskSize; j++) {
            printf("%0.4f ", mask[i * maskSize + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Aplica convolución secuencial a una imagen de entrada con la máscara dada
void convolutionHost(
    const unsigned char* input,
    float* output,
    int width, int height,
    int channels,
    const float* mask,
    int maskSize
) {
    int offset = maskSize / 2;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                float sum = 0.0f;
                for (int j = -offset; j <= offset; j++) {
                    for (int i = -offset; i <= offset; i++) {
                        int xi = x + i;
                        int yj = y + j;
                        if (xi >= 0 && xi < width && yj >= 0 && yj < height) {
                            int imageIndex = (yj * width + xi) * channels + c;
                            int maskIndex = (j + offset) * maskSize + (i + offset);
                            sum += input[imageIndex] * mask[maskIndex];
                        }
                    }
                }
                output[(y * width + x) * channels + c] = sum;
            }
        }
    }
}

// Genera imagen DoG (resta absoluta entre dos imágenes desenfocadas)
void generateDoGImage(const float* blur1, const float* blur2, float* dog, int size) {
    for (int i = 0; i < size; i++) {
        dog[i] = fabsf(blur1[i] - blur2[i]);
    }
}

// Calcula el error porcentual medio entre dos imágenes
double calculatePercentageError(const float* img1, const float* img2, int size) {
    double error_sum = 0.0;
    for (int i = 0; i < size; i++) {
        error_sum += fabs((double)img1[i] - (double)img2[i]);
    }
    double mean_error = error_sum / size;
    return (mean_error / 255.0) * 100.0;
}