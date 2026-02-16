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

  - **Multi-Precision Support:** Supports standard floating-point types (`float`, `double`) as well as NVIDIA's half and quarter precision types (`__half`, `__nv_bfloat16`, `__nv_fp8_e4m3`, `__nv_fp8_e5m2`) for GPU accelerated workloads.

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


## Quick Start & Building with Pixi

Locusta uses [Pixi](https://pixi.prefix.dev/) for cross-platform dependency management, task execution, and package building via the `pixi-build-cmake` backend.

### 1. Install Pixi
Follow the installation instructions at https://pixi.prefix.dev/.

### 2. Available Pixi Tasks

* **Configure project**:
  Configures the build directory using Ninja with Release optimizations, CUDA support, tests, and documentation enabled.
  ```bash
  pixi run configure
  ```

* **IDE / LSP Integration**:
  Symlinks `build/compile_commands.json` to the root directory for C++ IDEs and LSP engines (e.g., clangd):
  ```bash
  pixi run setup-ide
  ```

* **Build binaries**:
  Compiles all binaries (solvers, examples, and tests).
  ```bash
  pixi run build
  ```

* **Build Conda Package (via `pixi-build-cmake`)**:
  Locusta configures `pixi-build-cmake` as its package build backend. Build the binary package with:
  ```bash
  pixi build
  ```

* **Run CPU Benchmarks & Tests**:
  Automatically installs all toolchain dependencies, configures, builds, and executes CPU tests:
  ```bash
  pixi run locusta_test_cpu
  ```

* **Run CUDA Benchmarks & Tests**:
  If a CUDA-capable GPU and drivers are present:
  ```bash
  pixi run locusta_test_cuda
  ```

* **Build API Documentation**:
  Generates Doxygen documentation in `build/html/index.html`:
  ```bash
  pixi run doc
  ```

### 3. Test Filtering (GoogleTest)

You can filter test execution using standard GoogleTest `--gtest_filter` patterns:

```bash
# Run only Half Precision tests on GPU:
./build/bin/locustaExamples-cuda --gtest_filter='*Half*'

# Run only Particle Swarm Optimization tests on CPU:
./build/bin/locustaExamples-cpu --gtest_filter='*PSO*'
```

---

## Building directly with CMake

If you prefer to manage dependencies manually (without Pixi), you can build Locusta directly using CMake.

### Requirements

* CMake >= 3.5
* C++14 compliant compiler (GCC, Clang, MSVC)
* **Optional Dependencies**:
  * CUDA Toolkit >= 12.0 (for GPU acceleration)
  * Google Test (for unit tests and benchmarks)
  * OpenMP (for multi-threaded CPU execution)
  * Doxygen & Graphviz (for building API documentation)

### CMake Build Instructions

1. **Configure**:
   ```bash
   # Release build (default optimization flags: -O3 -DNDEBUG):
   cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON -DENABLE_TESTS=ON -DENABLE_DOCS=ON

   # Debug build (with debug symbols and no optimization: -O0 -ggdb -g -D_DEBUG):
   cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DENABLE_CUDA=ON -DENABLE_TESTS=ON -DENABLE_DOCS=ON
   ```
   *Configuration Flags:*
   - `-DCMAKE_BUILD_TYPE=Release/Debug/RelWithDebInfo`: Set build optimization level (`Debug` enables `-O0 -ggdb -g -D_DEBUG`, `Release` enables `-O3 -DNDEBUG`).
   - `-DENABLE_CUDA=ON/OFF`: Enable or disable CUDA GPU support (default: `ON`).
   - `-DENABLE_TESTS=ON/OFF`: Enable or disable building tests and examples (default: `ON`).
   - `-DENABLE_DOCS=ON/OFF`: Enable or disable Doxygen documentation target (default: `ON`).
   - `-DENABLE_SIMD=ON/OFF`: Enable SIMD vectorization flags (default: `ON`).

2. **Build**:
   ```bash
   cmake --build build -j
   ```

3. **Run Executables**:
   ```bash
   # Run CPU benchmark example tests:
   ./build/bin/locustaExamples-cpu

   # Run CUDA benchmark example tests (if ENABLE_CUDA=ON):
   ./build/bin/locustaExamples-cuda
   ```

4. **Build Documentation**:
   ```bash
   cmake --build build --target doc
   ```
   Output documentation is located at `build/html/index.html`.

## Implemented Metaheuristic Solvers & Precision Support

| Metaheuristic Solver           | OMP (CPU) | CUDA (GPU) | Precision Types Supported |
|--------------------------------|-----------|------------|---------------------------|
| Particle Swarm Optimization    | ✓         | ✓          | `float`, `double`, `__half`, `__nv_bfloat16`, `__nv_fp8_e4m3`, `__nv_fp8_e5m2` |
| Genetic Algorithm              | ✓         | ✓          | `float`, `double`, `__half`, `__nv_bfloat16`, `__nv_fp8_e4m3`, `__nv_fp8_e5m2` |
| Differential Evolution         | ✓         | ✓          | `float`, `double`, `__half`, `__nv_bfloat16`, `__nv_fp8_e4m3`, `__nv_fp8_e5m2` |

## License

Locusta is distributed under the terms of the GNU Lesser General Public License

See [LICENSE-LGPL-3.0](LICENSE-LGPL-3.0)
