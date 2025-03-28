# Locusta

![Locusta Logo - Designed by: Perla Fierro @ https://perlafierro.com/locusta](./locusta-06.png)

This repository contains the source code for the Locusta project.

It includes the library implementation, documentation, and an example benchmarking application.

For API details, see the Locusta Doxygen documentation at [rdelcueto.github.io/Locusta/](https://rdelcueto.github.io/Locusta/).

# Locusta - A Framework for Analyzing and Evaluating Evolutionary Algorithms on Parallel Architectures

## Introduction

### Overview

Locusta is a C++ header-only library that provides a flexible and efficient framework for implementing and running evolutionary algorithms on both CPUs and GPUs. It is designed to be massively parallel, taking advantage of the computational power of modern GPUs and multi-core CPUs to accelerate the optimization process.

This is the main source code repository for the Locusta project.
It contains the library code implementation, documentation and an example benchmarking application using the [GoogleTest library](https://github.com/google/googletest).

Locusta was developed as part of a dissertation exploring the performance characteristics of evolutionary algorithms on different parallel computing architectures. The research focused on comparing CPU and GPU implementations, identifying optimal strategies for each architecture, and analyzing the trade-offs involved.

### Thesis Abstract

The following abstract from the dissertation provides further context for the Locusta project:

> In the field of Computational Intelligence, various population-based metaheuristics have proven to be optimization methods that provide greater efficiency in a wide variety of applications. However, these algorithms require significant computational power to perform the evaluation of the objective function on each individual in the population, limiting their usefulness when sufficient computational resources are not available.
>
> The algorithms of some of these metaheuristics are inherently parallelizable, as is the evaluation of the objective function on the population set. Therefore, their algorithms can be ported to a wide variety of parallel computing architectures. Despite this, there are several details, both in the implementation and in the use cases of these metaheuristics, that will determine whether it is convenient to port the algorithms to a certain computing architecture.
>
> This thesis presents a comparative study on the parallelization on CPUs and GPUs of three bio-inspired metaheuristics: Particle Swarm Optimization, Genetic Algorithms, and Differential Evolution.
>
> For this work, a library was designed that implements each of the algorithms in a modular way, making it possible to use ad-hoc optimization strategies for each of the computing architectures and allowing analysis of the costs of each of the operations involved.

See the full dissertation for detailed information and analysis: [Parallelization of Bio-inspired Metaheuristics - Rodrigo Gonzalez del Cueto](https://hdl.handle.net/20.500.14330/TES01000763736)

## Key Features

  - **Header-only:** Easy to integrate into your projects.

  - **GPU-accelerated:** Significantly faster execution for large populations and complex fitness functions.

  - **CPU parallel:**  Provides a CPU parallel implementation using OpenMP, allowing for cross-platform comparison and performance evaluation.

  - **Flexible:** Supports various evolutionary algorithms, including:
    - Genetic algorithms
    - Particle swarm optimization
    - Differential evolution
    
  - **Extensible:** Allows users to define their own custom operators and fitness functions.

  - **Data-oriented design:**  Recognizes that each platform (CPU, GPU) presents unique data handling challenges, leading to specialized implementations for optimal performance.

## Getting Started

To get started with Locusta, include the necessary headers and create instances of the desired solver, evaluator, and population classes. Configure the solver with the desired parameters and operators.

## Examples

The Locusta library includes a set of usage examples and concrete implementations of the various components. The example files are located in the `example` directory.

* CPU Examples:
    * `benchmarks_cpu.hpp`: This file defines the `BenchmarkFunctor` class, which is a CPU implementation of the `EvaluationFunctor` interface.
    * `benchmarks_cpu_impl.hpp`: This file provides the implementations of various benchmark functions for the CPU.

* CUDA Examples:
    * `benchmarks_cuda.hpp`: This file defines the `BenchmarkCudaFunctor` class, which is a CUDA implementation of the `EvaluationCudaFunctor` interface.
    * `kernel/benchmarks_cuda.cu`: This file provides the CUDA kernels for evaluating the benchmark functions on the GPU.

## Architecture

Locusta employs a data-oriented design, recognizing that CPUs and GPUs have different strengths and weaknesses when it comes to data handling and computation. This means that the CPU and GPU implementations may differ significantly to take advantage of the respective architectures.

### GPU Architecture

The current implementation leverages CUDA to describe massively parallel kernels for:

* Evaluating the fitness function on a population of genomes
* Executing metaheuristic processes (e.g., mutation, crossover)
* Generating pseudo-random numbers

### CPU Architecture

The CPU parallel implementation is written using OpenMP, enabling efficient parallel computation on multi-core CPUs.

## Comparison and Performance

The framework allows for comparing and measuring the performance throughput of each implementation, taking into account the strengths of each targeted architecture. This helps users to choose the best platform for their specific needs and to optimize their evolutionary algorithms for maximum performance.


## Building

Instructions for building the example application and documentation.

### Building the Example Application

#### Ubuntu & WSL

1.  Install Dependencies
    *   CUDA & CUDA Toolkit. See [CUDA Quick start guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)
    *   CMake
        > sudo apt-get install cmake
    *   Google Test
        > sudo apt-get install libtgtest-dev

2.  Run Cmake
    > cmake -B build .

3.  Compile example application
    > cd Build
    > make -j

### Building the Documentation

#### Ubuntu

1.  Install Dependencies
    *   CMake
        > sudo apt-get install cmake
    *   Doxygen
        > sudo apt-get install doxygen
    *   Graphviz
        > sudo apt-get install graphviz

2.  Run Cmake
    > cmake -B build .

3.  Compile
    > cd Build
    > make doc

## Implemented Metaheuristic Solvers

| Metaheuristic Solver           | OMP |  CUDA |
|--------------------------------|-----|-------|
| Particle Swarm Optimization    | ✓   | ✓    |
| Genetic Algorithm              | ✓   | ✓    |
| Differential Evolution         | ✓   | ✓    |

## License

Locusta is distributed under the terms of the GNU Lesser General Public License

See [LICENSE-LGPL-3.0](LICENSE-LGPL-3.0)
