# Hybrid 3D GAN C and C++ Implementation

## Overview

This repository contains a pure C implementation of a 3D  deep learning model specidically a Hybrid GAN, with cross-attention, self-attention, and convolutional blocks in the generator. The self-attention, convolutional layers, and GAN structure, including a UNet architecture within the generator, are implemented in both C and Python.

## Features

- **Self-Attention:** Integrated into the generator and implemented in C and Python.
- **Cross-Attention Mechanism:** Designed for 3D tensors using PyTorch, applicable as input to CNN layers. This mechanism merges two separate input tensors, providing an output of the same size, which allows for multiple input images or the introduction of new data during network processing. Implemented in Python, C, and C++.
- **Convolutional Blocks:** Essential convolution operations for the GAN, implemented in C and Python.
- **GAN Structure:** The overall GAN architecture, including the use of a UNet within the generator, is implemented in C and Python.

## Current Development

We are currently building the training script in C, including many dependencies typically found in the PyTorch library, from scratch.

## To-Do List

- Implement GAN training and code iteration in C and C++.
- Update the README with detailed instructions and examples.
- Optimize nested for loops with arithmetic operations to save memory.

## Getting Started

### Prerequisites

- C Compiler (e.g., GCC)
- Python (for the Python implementation and cross-attention mechanism)
- PyTorch (for the cross-attention mechanism in Python)
- C++ Compiler (for the C++ implementation)

### Building the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/Anne-Andresen/hybrid-3d-gan.git
   cd hybrid-3d-gan
   ```

2. Compile the C code:
``` bash
   
   gcc -o hybrid_gan src/hybrid_gan.c -lm
```

## Usage

- C Implementation: Run the compiled C code:
``` bash
./hybrid_gan

```
- Python Implementation: Execute the Python script:
``` bash
python src/hybrid_gan.py


```

## Contributing


Contributions are welcome. Please feel free to submit pull requests or open issues with suggestions and improvements.

## License


This project is licensed under the MIT License - see the LICENSE file for details.
