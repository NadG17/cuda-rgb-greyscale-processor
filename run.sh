#!/bin/bash

# run.sh - Automated execution script for CUDA RGB to Greyscale Image Processor
# This script handles the complete workflow from compilation to execution

set -e  # Exit on any error

echo "=============================================="
echo "CUDA RGB to Greyscale Image Processor"
echo "=============================================="

# Function to print colored output
print_status() {
    # Compile the program
    nvcc $NVCC_FLAGS -o convertRGBToGrey convertRGBToGrey.cu $OPENCV_FLAGS
    
    if [ $? -eq 0 ]; then
        print_status "Compilation successful"
    else
        print_error "Compilation failed"
        exit 1
    fi
}

# Run the program with different configurations
run_tests() {
    print_status "Running image processing tests..."
    
    # Create output directory
    mkdir -p random_greyscaled_images
    mkdir -p comparison_images
    
    echo ""
    echo "=== Test 1: Basic Processing ==="
    ./convertRGBToGrey --input random_color_images --output random_greyscaled_images
    
    echo ""
    echo "=== Test 2: Optimized Kernel ==="
    ./convertRGBToGrey --input random_color_images --output random_greyscaled_images --optimized
    
    echo ""
    echo "=== Test 3: Single Image with Benchmark ==="
    if [ -f "random_color_images/large_test.jpg" ]; then
        ./convertRGBToGrey --input random_color_images/large_test.jpg --output comparison_images/large_test_greyscale.jpg --benchmark --optimized
    fi
    
    echo ""
    echo "=== Test 4: Performance Comparison ==="
    ./convertRGBToGrey --input random_color_images --output comparison_images --benchmark --optimized
}

# Create execution log
create_log() {
    print_status "Creating execution log..."
    
    LOG_FILE="execution_log.txt"
    
    {
        echo "CUDA RGB to Greyscale Image Processor - Execution Log"
        echo "======================================================"
        echo "Execution Date: $(date)"
        echo "Environment: $(if [ $IN_COLAB -eq 1 ]; then echo 'Google Colab'; else echo 'Local System'; fi)"
        echo ""
        
        echo "System Information:"
        echo "==================="
        if command -v nvidia-smi &> /dev/null; then
            echo "GPU Information:"
            nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
            echo ""
        fi
        
        echo "CUDA Version:"
        nvcc --version | grep "release"
        echo ""
        
        echo "OpenCV Version:"
        pkg-config --modversion opencv4 2>/dev/null || pkg-config --modversion opencv 2>/dev/null || echo "Version unknown"
        echo ""
        
        echo "Processed Images:"
        echo "=================="
        if [ -d "random_greyscaled_images" ]; then
            echo "Output images in random_greyscaled_images/:"
            ls -la random_greyscaled_images/ | grep -E '\.(jpg|png|bmp)echo -e "\033[1;32m[INFO]\033[0m $1"
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
    
     | wc -l | xargs echo "Total processed images:"
            echo ""
            echo "Sample output files:"
            ls random_greyscaled_images/ | head -10
        fi
        
        if [ -d "comparison_images" ]; then
            echo ""
            echo "Comparison images in comparison_images/:"
            ls -la comparison_images/
        fi
        
        echo ""
        echo "File sizes comparison:"
        echo "======================"
        if [ -d "random_color_images" ] && [ -d "random_greyscaled_images" ]; then
            echo "Input directory size:"
            du -sh random_color_images/
            echo "Output directory size:"
            du -sh random_greyscaled_images/
        fi
        
    } > $LOG_FILE
    
    print_status "Execution log saved to $LOG_FILE"
}

# Create a performance report
create_performance_report() {
    print_status "Creating performance report..."
    
    REPORT_FILE="performance_report.md"
    
    cat > $REPORT_FILE << EOF
# CUDA RGB to Greyscale Conversion - Performance Report

## System Configuration
- **Date**: $(date)
- **Environment**: $(if [ $IN_COLAB -eq 1 ]; then echo 'Google Colab'; else echo 'Local System'; fi)
- **CUDA Version**: $(nvcc --version | grep "release" | cut -d' ' -f5,6)
- **GPU**: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
- **GPU Memory**: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)

## Test Results

### Image Processing Summary
- **Total Images Processed**: $(ls random_greyscaled_images/*.jpg 2>/dev/null | wc -l)
- **Input Directory Size**: $(du -sh random_color_images/ 2>/dev/null | cut -f1)
- **Output Directory Size**: $(du -sh random_greyscaled_images/ 2>/dev/null | cut -f1)

### Sample Outputs
The following images demonstrate the RGB to greyscale conversion:

$(for img in random_greyscaled_images/*.jpg; do
    if [ -f "$img" ]; then
        echo "- $(basename "$img")"
    fi
done | head -5)

### Performance Characteristics
- **Kernel Type**: Both standard and optimized kernels implemented
- **Memory Management**: Efficient GPU memory allocation and deallocation
- **Error Handling**: Comprehensive CUDA error checking
- **Scalability**: Processes both individual images and batch directories

## Technical Implementation
- **Block Size**: 16x16 threads per block
- **Memory Access**: Coalesced memory access patterns
- **Optimization**: Shared memory utilization in optimized kernel
- **Compatibility**: Multiple CUDA compute capabilities supported

## Verification
All processed images can be found in the \`random_greyscaled_images/\` directory.
Execution details are available in \`execution_log.txt\`.
EOF
    
    print_status "Performance report saved to $REPORT_FILE"
}

# Main execution function
main() {
    echo "Starting CUDA RGB to Greyscale Image Processor..."
    echo ""
    
    # Check environment
    check_colab
    
    # Install dependencies if in Colab
    if [ $IN_COLAB -eq 1 ]; then
        install_colab_dependencies
    fi
    
    # Check CUDA
    check_cuda
    
    # Generate test images
    generate_test_images
    
    # Compile program
    compile_program
    
    # Run tests
    run_tests
    
    # Create documentation
    create_log
    create_performance_report
    
    print_status "All tasks completed successfully!"
    echo ""
    echo "Output files:"
    echo "- Processed images: random_greyscaled_images/"
    echo "- Comparison images: comparison_images/"
    echo "- Execution log: execution_log.txt"
    echo "- Performance report: performance_report.md"
    echo ""
    echo "To run individual tests:"
    echo "  ./convertRGBToGrey --help"
}

# Parse command line arguments
case "${1:-run}" in
    "install-deps")
        if check_colab; then
            install_colab_dependencies
        else
            echo "For local installation, use: make install-deps"
        fi
        ;;
    "compile")
        compile_program
        ;;
    "generate")
        generate_test_images
        ;;
    "test")
        run_tests
        ;;
    "clean")
        print_status "Cleaning up..."
        rm -f convertRGBToGrey
        rm -rf random_greyscaled_images comparison_images
        rm -f execution_log.txt performance_report.md
        print_status "Cleanup complete"
        ;;
    "help")
        echo "Usage: $0 [command]"
        echo "Commands:"
        echo "  run (default)  - Complete workflow: install, compile, generate, test"
        echo "  install-deps   - Install dependencies"
        echo "  compile        - Compile the CUDA program"
        echo "  generate       - Generate test images"
        echo "  test           - Run processing tests"
        echo "  clean          - Clean up generated files"
        echo "  help           - Show this help"
        ;;
    "run"|*)
        main
        ;;
esacecho -e "\033[1;32m[INFO]\033[0m $1"
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
