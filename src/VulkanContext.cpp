#include "VulkanContext.hpp"
#include <stdexcept>
#include <vector>
#include <fstream>
#include <cstring>

// Helper function to read shader files
std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file!");
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0); 
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}

VulkanContext::VulkanContext(int matrix_size) : matrixSize(matrix_size) {
    std::cout << "Initializing Vulkan Context..." << std::endl;

    createInstance();
    pickPhysicalDevice();
    createLogicalDevice();
    createCommandPool();
    createDescriptorSetLayout();
    createComputePipeline();
    createDescriptorPool();

    std::cout << "Vulkan Context initialized successfully." << std::endl;
}

VulkanContext::~VulkanContext() {
    cleanup();
    std::cout << "Vulkan Context destroyed." << std::endl;
}

void VulkanContext::createInstance() {
    std::cout << "Creating Vulkan Instance..." << std::endl;

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Matrix Multiply";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan instance!");
    }

    std::cout << "Vulkan Instance created successfully." << std::endl;
}

void VulkanContext::pickPhysicalDevice() {
    std::cout << "Selecting a Physical Device (GPU)..." << std::endl;

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0) {
        throw std::runtime_error("Failed to find GPUs with Vulkan support!");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    // Pick the first device with compute capability
    for (const auto& device : devices) {
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(device, &deviceProperties);
        std::cout << "Found device: " << deviceProperties.deviceName << std::endl;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        for (uint32_t i = 0; i < queueFamilyCount; ++i) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                physicalDevice = device;
                computeQueueFamilyIndex = i;
                std::cout << "Selected device: " << deviceProperties.deviceName << std::endl;
                return;
            }
        }
    }

    throw std::runtime_error("Failed to find a suitable GPU with compute capabilities!");
}

void VulkanContext::createLogicalDevice() {
    std::cout << "Creating Logical Device..." << std::endl;

    float queuePriority = 1.0f;

    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = computeQueueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkPhysicalDeviceFeatures deviceFeatures{};

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.queueCreateInfoCount = 1;
    createInfo.pEnabledFeatures = &deviceFeatures;

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create logical device!");
    }

    vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &computeQueue);

    std::cout << "Logical Device created successfully." << std::endl;
}

void VulkanContext::createCommandPool() {
    std::cout << "Creating Command Pool..." << std::endl;

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = computeQueueFamilyIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool!");
    }

    std::cout << "Command Pool created successfully." << std::endl;
}

void VulkanContext::createDescriptorSetLayout() {
    std::cout << "Creating Descriptor Set Layout..." << std::endl;

    std::vector<VkDescriptorSetLayoutBinding> bindings(3);

    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout!");
    }

    std::cout << "Descriptor Set Layout created successfully." << std::endl;
}

void VulkanContext::createComputePipeline() {
    std::cout << "Creating Compute Pipeline..." << std::endl;

    auto shaderCode = readFile("shaders/comp.spv");

    VkShaderModuleCreateInfo shaderModuleInfo{};
    shaderModuleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleInfo.codeSize = shaderCode.size();
    shaderModuleInfo.pCode = reinterpret_cast<const uint32_t*>(shaderCode.data());

    if (vkCreateShaderModule(device, &shaderModuleInfo, nullptr, &computeShaderModule) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module!");
    }

    VkPipelineShaderStageCreateInfo shaderStageInfo{};
    shaderStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = computeShaderModule;
    shaderStageInfo.pName  = "main";

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount         = 1;
    pipelineLayoutInfo.pSetLayouts            = &descriptorSetLayout;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout!");
    }

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage  = shaderStageInfo;
    pipelineInfo.layout = pipelineLayout;

    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create compute pipeline!");
    }

    std::cout << "Compute Pipeline created successfully." << std::endl;
}

void VulkanContext::createDescriptorPool() {
    std::cout << "Creating Descriptor Pool..." << std::endl;

    std::vector<VkDescriptorPoolSize> poolSizes(1);
    poolSizes[0].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = 3;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes    = poolSizes.data();
    poolInfo.maxSets       = 1;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool!");
    }

    std::cout << "Descriptor Pool created successfully." << std::endl;
}

void VulkanContext::createDescriptorSets() {
    std::cout << "Allocating and Updating Descriptor Sets..." << std::endl;

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts        = &descriptorSetLayout;

    if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor set!");
    }

    VkDescriptorBufferInfo bufferInfoA{};
    bufferInfoA.buffer = bufferA;
    bufferInfoA.offset = 0;
    bufferInfoA.range  = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo bufferInfoB{};
    bufferInfoB.buffer = bufferB;
    bufferInfoB.offset = 0;
    bufferInfoB.range  = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo bufferInfoC{};
    bufferInfoC.buffer = bufferC;
    bufferInfoC.offset = 0;
    bufferInfoC.range  = VK_WHOLE_SIZE;

    std::vector<VkWriteDescriptorSet> descriptorWrites(3);

    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = descriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &bufferInfoA;

    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = descriptorSet;
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pBufferInfo = &bufferInfoB;

    descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[2].dstSet = descriptorSet;
    descriptorWrites[2].dstBinding = 2;
    descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[2].descriptorCount = 1;
    descriptorWrites[2].pBufferInfo = &bufferInfoC;

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);

    std::cout << "Descriptor Sets allocated and updated successfully." << std::endl;
}

void VulkanContext::createBuffers(float* matrixA, float* matrixB, float* matrixC, int matrix_size) {
    std::cout << "Creating Buffers for Matrices..." << std::endl;

    VkDeviceSize bufferSize = sizeof(float) * matrix_size * matrix_size;

    VkBufferCreateInfo bufferInfoA{};
    bufferInfoA.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfoA.size = bufferSize;
    bufferInfoA.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfoA.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfoA, nullptr, &bufferA) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer A!");
    }

    allocateBufferMemory(bufferA, &bufferMemoryA, bufferSize, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkBufferCreateInfo bufferInfoB{};
    bufferInfoB.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfoB.size = bufferSize;
    bufferInfoB.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfoB.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfoB, nullptr, &bufferB) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer B!");
    }

    allocateBufferMemory(bufferB, &bufferMemoryB, bufferSize, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VkBufferCreateInfo bufferInfoC{};
    bufferInfoC.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfoC.size = bufferSize;
    bufferInfoC.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfoC.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfoC, nullptr, &bufferC) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer C!");
    }

    allocateBufferMemory(bufferC, &bufferMemoryC, bufferSize, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    std::cout << "Buffers created successfully." << std::endl;

    createDescriptorSets();
}

void VulkanContext::allocateBufferMemory(VkBuffer buffer, VkDeviceMemory* bufferMemory, VkDeviceSize size, VkMemoryPropertyFlags properties) {
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, bufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device, buffer, *bufferMemory, 0);
}

uint32_t VulkanContext::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type!");
}

void VulkanContext::runComputeShader(float* matrixA, float* matrixB, float* matrixC, int matrix_size) {
    std::cout << "Running Compute Shader..." << std::endl;

    createBuffers(matrixA, matrixB, matrixC, matrix_size);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffer!");
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin command buffer!");
    }

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

    uint32_t groupCountX = (matrixSize + 15) / 16;
    uint32_t groupCountY = (matrixSize + 15) / 16;

    vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to record command buffer!");
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    VkFence fence;
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create fence!");
    }

    if (vkQueueSubmit(computeQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit compute command buffer!");
    }

    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

    vkDestroyFence(device, fence, nullptr);

    void* data;
    vkMapMemory(device, bufferMemoryC, 0, sizeof(float) * matrixSize * matrixSize, 0, &data);
    memcpy(matrixC, data, sizeof(float) * matrixSize * matrixSize);
    vkUnmapMemory(device, bufferMemoryC);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);

    std::cout << "Compute Shader completed successfully." << std::endl;
}

void VulkanContext::cleanup() {
    std::cout << "Cleaning up Vulkan resources..." << std::endl;

    vkDestroyBuffer(device, bufferA, nullptr);
    vkDestroyBuffer(device, bufferB, nullptr);
    vkDestroyBuffer(device, bufferC, nullptr);
    vkFreeMemory(device, bufferMemoryA, nullptr);
    vkFreeMemory(device, bufferMemoryB, nullptr);
    vkFreeMemory(device, bufferMemoryC, nullptr);
    vkDestroyPipeline(device, computePipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    vkDestroyShaderModule(device, computeShaderModule, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    std::cout << "Vulkan resources cleaned up successfully." << std::endl;
}

