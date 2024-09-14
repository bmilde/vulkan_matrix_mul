cmake_minimum_required(VERSION 3.10)
project(vulkan_matrix_mul)

find_package(Vulkan REQUIRED)

set(SOURCES
    src/main.cpp
    src/VulkanContext.cpp
)

add_executable(vulkan_matrix_mul ${SOURCES})

set(CMAKE_BUILD_TYPE Debug) # Ensure debug build
set(CMAKE_CXX_FLAGS_DEBUG "-g") # Add debug symbols

target_include_directories(vulkan_matrix_mul PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries(vulkan_matrix_mul ${Vulkan_LIBRARIES})
