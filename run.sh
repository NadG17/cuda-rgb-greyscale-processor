#!/bin/bash

# run.sh - Automated execution script for CUDA RGB to Greyscale Image Processor
# This script handles the complete workflow from compilation to execution

set -e  # Exit on any error

echo "=============================================="
echo "CUDA RGB to Greyscale Image Processor"
echo "=============================================="

# Function to print colored output
print_status() {
    echo -e "\033[1;32m[INFO]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

# Check if we're running in Google Colab
check_colab() {
    if [ -d "/content" ] && [ -f "/usr/local/cuda/bin/nvcc" ]; then
        export IN_COLAB=1
        print_status "Detected Google Colab environment"
        return 0
    else
        export IN_COLAB=0
        return 1
    fi
}

# Install dependencies for Colab
install_colab_dependencies() {
    print_status "Installing dependencies for Google Colab..."
    
    # Update package list
    apt-get update -qq
    
    # Install build tools
    apt-get install -y build-essential
    
    # Install OpenCV development libraries
    apt-get install -y libopencv-dev
    
    # Install Python packages
    pip install opencv-python numpy Pillow
    
    # Set CUDA paths
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    
    print_status "Dependencies installed successfully"
}

# Check CUDA availability
check_cuda() {
    print_status "Checking CUDA installation..."
    
    if ! command -v nvcc &> /dev/null; then
        print_error "NVCC not found. Please install CUDA toolkit."
        exit 1
    fi
    
    if ! nvidia-smi &> /dev/null; then
        print_error "nvidia-smi not found or no GPU available."
        exit 1
    fi
    
    echo "CUDA Version:"
    nvcc --version | grep "release"
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
}

# Generate synthetic test images
generate_test_images() {
    print_status "Generating synthetic test images..."
    
    mkdir -p random_color_images
    
    python3 << 'EOF'
import cv2
import numpy as np
import os

# Create directory if it doesn't exist
os.makedirs('random_color_images', exist_ok=True)

# Generate various types of test images
print("Generating test images...")

# 1. Random noise images
for i in range(10):
    img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cv2.imwrite(f'random_color_images/random_{i:03d}.jpg', img)

# 2. Gradient images
for i in range(5):
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    for x in range(512):
        for y in range(512):
            img[y, x, 0] = int(255 * x / 512)  # Red gradient
            img[y, x, 1] = int(255 * y / 512)  # Green gradient  
            img[y, x, 2] = int(255 * (x + y) / 1024)  # Blue gradient
    cv2.imwrite(f'random_color_images/gradient_{i:03d}.jpg', img)

# 3. Geometric patterns
for i in range(5):
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    center = (256, 256)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    for j in range(5):
        cv2.circle(img, center, 50 + j * 30, colors[j], -1)
    cv2.imwrite(f'random_color_images/circles_{i:03d}.jpg', img)

# 4. Large images for performance testing
large_img = np.random.randint(0, 256, (2048, 2048, 3), dtype=np.uint8)
cv2.imwrite('random_color_images/large_test.jpg', large_img)

print(f"Generated 21 test images in random_color_images/")
EOF
    
    print_status "Test images generated successfully"
}

# Compile the CUDA program
compile_program() {
    print_status "Compiling CUDA program..."
    
    # Set compiler flags
    NVCC_FLAGS="-std=c++17 -O3 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80"
    
    # Get OpenCV flags
    if pkg-config --exists opencv4; then
        OPENCV_FLAGS=$(pkg-config --cflags --libs opencv4)
    elif pkg-config --exists opencv; then
        OPENCV_FLAGS=$(pkg-config --cflags --libs opencv)
    else
        print_warning "pkg-config for OpenCV not found, using default flags"
        OPENCV_FLAGS="-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui"
    fi
