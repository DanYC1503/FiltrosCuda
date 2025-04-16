#include <iostream>
#include <string>
#include <vector>
#include <chrono> 

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "functions.h"



int main() {
    int width, height, channels;
    std::string imagePath = "C:/Users/danie/Downloads/test.jpg";
    unsigned char* h_inputImage = stbi_load(imagePath.c_str(), &width, &height, &channels, 3);

    if (!h_inputImage) {
        std::cerr << "Error loading image: " << imagePath << "\n";
        return 1;
    }

    std::cout << "Image loaded successfully!\n";
    std::cout << "Width: " << width << ", Height: " << height << ", Channels: " << channels << "\n";

    int kernelSize = 50;
    std::vector<float> motionBlurKernel = generateMotionBlurKernel(kernelSize);

    size_t img_size = width * height * channels;
    size_t img_bytes = img_size * sizeof(unsigned char);
    size_t img_floats = img_size * sizeof(float);

    // Reservar memoria
    unsigned char* h_outputSequential = new unsigned char[img_bytes];
    unsigned char* h_outputMeanCPU = new unsigned char[img_bytes];
    unsigned char* h_outputParallel = new unsigned char[img_bytes];
    unsigned char* h_outputMeanGPU = new unsigned char[img_bytes];
    /*
    // --- Secuencial con medición de tiempo ---
    auto startSeq = std::chrono::high_resolution_clock::now();

    // Filtro convolución secuencial
    auto startConv = std::chrono::high_resolution_clock::now();
    applyConvolutionSequentialRGB(h_inputImage, h_outputSequential, width, height, motionBlurKernel, kernelSize);
    auto endConv = std::chrono::high_resolution_clock::now();
    std::cout << "Convolución (CPU) tardó: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(endConv - startConv).count() << " ms\n";

    // Filtro promedio secuencial
    auto startMean = std::chrono::high_resolution_clock::now();
    mean_filter_cpu(h_inputImage, h_outputMeanCPU, width, height, channels, kernelSize, motionBlurKernel);
    auto endMean = std::chrono::high_resolution_clock::now();
    std::cout << "Filtro promedio (CPU) tardó: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(endMean - startMean).count() << " ms\n";

    auto endSeq = std::chrono::high_resolution_clock::now();
    std::cout << "Total procesamiento secuencial (CPU) tardó: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(endSeq - startSeq).count() << " ms\n";

    stbi_write_png("output_sequential_13x13.png", width, height, 3, h_outputSequential, width * 3);
    stbi_write_png("output_mean_cpu_13x13.png", width, height, 3, h_outputMeanCPU, width * 3);

    // --- DoG Secuencial ---
    float sigma1 = 1.0f, sigma2 = 2.0f;

    float* mask1 = new float[kernelSize * kernelSize];
    float* mask2 = new float[kernelSize * kernelSize];
    float* blur1 = new float[img_size];
    float* blur2 = new float[img_size];
    float* dog = new float[img_size];
    unsigned char* dog_output = new unsigned char[img_size];

    createGaussianMask(mask1, kernelSize, sigma1);
    createGaussianMask(mask2, kernelSize, sigma2);

    convolutionHost(h_inputImage, blur1, width, height, channels, mask1, kernelSize);
    convolutionHost(h_inputImage, blur2, width, height, channels, mask2, kernelSize);

    generateDoGImage(blur1, blur2, dog, img_size);

    // Convertir DoG a unsigned char para guardar como imagen
    for (size_t i = 0; i < img_size; ++i) {
        float val = dog[i];
        dog_output[i] = static_cast<unsigned char>(std::min(std::max(val, 0.0f), 255.0f));
    }

    stbi_write_png("output_dog_13x13.png", width, height, 3, dog_output, width * 3);
    std::cout << "DoG generado y guardado como output_dog_13x13.png\n";
    */
    // --- Paralelo (GPU) ---
    applyConvolutionParallelRGB(h_inputImage, h_outputMeanGPU, h_outputParallel, width, height, channels, motionBlurKernel, kernelSize);

    // Cleanup
    stbi_image_free(h_inputImage);
    delete[] h_outputSequential;
    delete[] h_outputMeanCPU;
    delete[] h_outputParallel;
    delete[] h_outputMeanGPU;

    return 0;
}



