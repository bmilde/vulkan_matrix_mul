#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include "VulkanContext.hpp"

const int MATRIX_SIZE = 4096;
const int NUM_ITERATIONS = 1000;

// Function to generate a random float
float randomFloat() {
    static std::default_random_engine generator;
    static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    return distribution(generator);
}

int main() {
    try {
        std::cout << "Starting Vulkan Matrix Multiplication..." << std::endl;

        // Initialize Vulkan context
        VulkanContext context(MATRIX_SIZE);

        // Create and initialize matrices
        std::vector<float> matrixA(MATRIX_SIZE * MATRIX_SIZE);
        std::vector<float> matrixB(MATRIX_SIZE * MATRIX_SIZE);
        std::vector<float> matrixC(MATRIX_SIZE * MATRIX_SIZE, 0.0f); // Result matrix

        std::cout << "Generating random matrices..." << std::endl;

        // Fill matrices A and B with random data
        for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i) {
            matrixA[i] = randomFloat();
            matrixB[i] = randomFloat();
        }

        std::cout << "Matrices generated. Starting computation..." << std::endl;

        // Start timing
        auto start = std::chrono::high_resolution_clock::now();

        for (int iteration = 0; iteration < NUM_ITERATIONS; ++iteration) {
            std::cout << "Iteration " << (iteration + 1) << " of " << NUM_ITERATIONS << "..." << std::endl;
            context.runComputeShader(matrixA.data(), matrixB.data(), matrixC.data(), MATRIX_SIZE);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        std::cout << "Computation completed in " << elapsed.count() << " seconds." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
