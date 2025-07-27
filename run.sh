#!/bin/bash

print_info() {
    echo -e "\033[1;32m[INFO]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

check_cuda() {
    if ! command -v nvcc &> /dev/null; then
        print_error "CUDA (nvcc) not found! Please install the CUDA Toolkit and ensure 'nvcc' is in your PATH."
        exit 1
    fi
}

check_make() {
    if ! command -v make &> /dev/null; then
        print_error "'make' command not found! Please install make."
        exit 1
    fi
}

check_image_file() {
    if [ ! -f "$1" ]; then
        print_error "Input image file '$1' does not exist!"
        exit 1
    fi
}

clean_project() {
    print_info "Cleaning project..."
    rm -rf bin
    rm -rf build
    rm -f cuda_image_processor
    print_info "Cleanup complete."
}

build_project() {
    print_info "Building CUDA Image Processor..."
    mkdir -p build
    mkdir -p bin
    nvcc -std=c++17 -O3 -arch=sm_75 -Xptxas -O3 -use_fast_math src/main.cu src/process.cu src/utils.cu -o cuda_image_processor -lcuda -lcudart -lstdc++fs
    if [ $? -ne 0 ]; then
        print_error "Compilation failed."
        exit 1
    fi
    print_info "Build complete. Executable: ./cuda_image_processor"
}

run_test_image() {
    if [ ! -f "input.png" ]; then
        print_warning "No test image found. Downloading sample input.png..."
        curl -L -o input.png https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png
    fi

    build_project

    print_info "Running CUDA Image Processor on sample image..."
    ./cuda_image_processor input.png output.png test 256

    if [ $? -ne 0 ]; then
        print_error "Execution failed."
        exit 1
    fi

    print_info "Processed Images:"
    echo "=================="
    if [ -d "random_greyscaled_images" ]; then
        echo "Output images in random_greyscaled_images/:"
        ls -la random_greyscaled_images/ | grep -E '\.(jpg|png|bmp)$'
    fi
}

run_custom_image() {
    input_image=$1
    output_image=$2
    part=$3
    threads=$4

    check_image_file "$input_image"
    build_project

    print_info "Running CUDA Image Processor..."
    ./cuda_image_processor "$input_image" "$output_image" "$part" "$threads"

    if [ $? -ne 0 ]; then
        print_error "Execution failed."
        exit 1
    fi

    print_info "Output saved to: $output_image"
}

show_help() {
    echo "Usage:"
    echo "  ./run.sh test                    # Run on default test image"
    echo "  ./run.sh clean                   # Clean project"
    echo "  ./run.sh build                   # Build project"
    echo "  ./run.sh run <in> <out> <part> <threads>   # Run with custom image"
    echo ""
    echo "Example:"
    echo "  ./run.sh run input.jpg result.png test 256"
}

main() {
    if [ "$#" -lt 1 ]; then
        show_help
        exit 1
    fi

    command=$1
    case $command in
        test)
            check_cuda
            check_make
            run_test_image
            ;;
        build)
            check_cuda
            check_make
            build_project
            ;;
        clean)
            clean_project
            ;;
        run)
            if [ "$#" -ne 5 ]; then
                print_error "Invalid number of arguments for 'run'."
                show_help
                exit 1
            fi
            check_cuda
            check_make
            run_custom_image "$2" "$3" "$4" "$5"
            ;;
        *)
            print_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
