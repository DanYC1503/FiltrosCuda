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

    // Cargar imagen RGB
    std::string imagePath = "C:/Users/danie/Downloads/test.jpg";
    unsigned char* h_inputImage = stbi_load(imagePath.c_str(), &width, &height, &channels, 3); // Num canales 3 = RGB

    if (!h_inputImage) {
        std::cerr << "Error loading image: " << imagePath << "\n";
        return 1;
    }

    std::cout << "Image loaded successfully!\n";
    std::cout << "Width: " << width << ", Height: " << height << ", Channels: " << channels << "\n";

    // Tamaño del kernel
    int kernelSize = 81;
    std::vector<float> motionBlurKernel = generateMotionBlurKernel(kernelSize);

    // Reservar memoria para salida
    //unsigned char* h_outputSequential = new unsigned char[width * height * 3];
    unsigned char* h_outputParallel = new unsigned char[width * height * 3];

    // Medir tiempo secuencial
    //auto startSeq = std::chrono::high_resolution_clock::now();
    //applyConvolutionSequentialRGB(h_inputImage, h_outputSequential, width, height, motionBlurKernel, kernelSize);
    //auto endSeq = std::chrono::high_resolution_clock::now();
    //auto durationSeq = std::chrono::duration_cast<std::chrono::milliseconds>(endSeq - startSeq).count();

    //stbi_write_png("output_sequential.png", width, height, 3, h_outputSequential, width * 3);
    //std::cout << "Secuencial tardó: " << durationSeq << " ms\n";

    // Medir tiempo paralelo
    auto startPar = std::chrono::high_resolution_clock::now();
    applyConvolutionParallelRGB(h_inputImage, h_outputParallel, width, height, motionBlurKernel, kernelSize);
    auto endPar = std::chrono::high_resolution_clock::now();
    auto durationPar = std::chrono::duration_cast<std::chrono::milliseconds>(endPar - startPar).count();

    stbi_write_png("output_parallel.png", width, height, 3, h_outputParallel, width * 3);
    std::cout << "Paralelo (GPU) tardó: " << durationPar << " ms\n";

    // Cleanup
    stbi_image_free(h_inputImage);
    //delete[] h_outputSequential;
    delete[] h_outputParallel;

    return 0;
}
