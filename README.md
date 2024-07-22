# Multi-Modal-Cuda-C-GAN

## Overview

Welcome to the Multi-Modal-Cuda-C-GAN repository! This project on a 3D deep learning model implemented in CUDA/C, specifically a Hybrid GAN that integrates cross-attention, self-attention, and convolutional blocks within the generator. The model leverages C for high-performance and scalable deep learning solutions.

## Features

- **Self-Attention**: Enhances the generator's capability, implemented in C for flexibility.
- **Cross-Attention Mechanism**: Designed for 3D tensors, suitable for CNN layers. It merges separate input tensors, outputting the same size, facilitating multi-input images and new data introduction during processing. Available in C and C++.
- **Convolutional Blocks**: Core convolution operations for the GAN, implemented in C for efficiency.
- **GAN Structure**: Comprehensive GAN architecture featuring a UNet within the generator, implemented in C for robust performance.

## Current Development

We are actively developing the training script in C, recreating many dependencies typically found in PyTorch from scratch, to ensure optimal performance and customization.

## To-Do List

- Implement GAN training and code iteration in C.
- Update the README with detailed setup instructions and usage examples.
- Optimize nested for loops and arithmetic operations for memory efficiency.

## Getting Started

### Prerequisites

- **C Compiler**: GCC or equivalent

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/Multi-Modal-Cuda-C-GAN.git
    cd Multi-Modal-Cuda-C-GAN
    ```

2. Compile the code:
    ```sh
    gcc -o main main.c -lcuda
    ```




## Contributing

We welcome contributions! Please check the [issues](https://github.com/Anne-Andresen/Multi-Modal-Cuda-C-GAN/issues) for tasks that need assistance or open a new issue to propose enhancements and report bugs.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/Anne-Andresen/Multi-Modal-Cuda-C-GAN/LICENSE) file for details.

## Contact

For any questions or feedback, please contact [aha.andresen@gmail.com](mailto:your-email@example.com).

