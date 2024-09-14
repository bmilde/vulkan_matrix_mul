#ifndef VULKAN_CONTEXT_HPP
#define VULKAN_CONTEXT_HPP

#include <vulkan/vulkan.h>
#include <vector>
#include <iostream>

class VulkanContext {
public:
    VulkanContext(int matrix_size);
    ~VulkanContext();

    void runComputeShader(float* matrixA, float* matrixB, float* matrixC, int matrix_size);

private:
    void createInstance();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createCommandPool();
    void createComputePipeline();
    void createDescriptorSetLayout();
    void createDescriptorPool();
    void createDescriptorSets();
    void createBuffers(float* matrixA, float* matrixB, float* matrixC, int matrix_size);
    void allocateBufferMemory(VkBuffer buffer, VkDeviceMemory* bufferMemory, VkDeviceSize size, VkMemoryPropertyFlags properties);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    void recordCommandBuffer();
    void cleanup();

    VkInstance instance;
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    VkQueue computeQueue;
    uint32_t computeQueueFamilyIndex;
    VkCommandPool commandPool;
    VkPipeline computePipeline;
    VkPipelineLayout pipelineLayout;
    VkShaderModule computeShaderModule;
    VkCommandBuffer commandBuffer;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;

    VkBuffer bufferA, bufferB, bufferC;
    VkDeviceMemory bufferMemoryA, bufferMemoryB, bufferMemoryC;

    int matrixSize;
};

#endif
