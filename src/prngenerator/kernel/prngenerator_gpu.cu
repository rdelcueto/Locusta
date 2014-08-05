#include <iostream>

#include "../prngenerator_gpu.h"

#include "cuda_common/cuPrintf.cu"

namespace locusta {

    template<typename TFloat>
    __global__ void
    gpu_setup_curand(uint64_t seed,
                     curandState *curand_states,
                     uint32_t num_generators)
    {
        uint64_t id =
            blockIdx.y * gridDim.x * blockDim.x +
            blockIdx.x * blockDim.x +
            threadIdx.x;

        while(id < num_generators)
        {
            curand_init(id + seed, 0, 0, &curand_states[id]);

#ifdef _DEBUG
            float x = curand_uniform(&curand_states[id]);
            CUPRINTF("%d): %f\n", id, x);
#endif
        
            id += blockDim.x * gridDim.x + blockDim.y * blockDim.y;
        }
    }

    template<typename TFloat>
    void gpu_setup_curand_wrapper(uint64_t seed,
                                  curandState *curand_states,
                                  uint32_t num_generators)
    {
#ifdef _DEBUG
        std::cout << "INIT Kernel Printing ...\n";
        std::cout << "Generators: " << num_generators << std::endl;
        cudaPrintfInit();
#endif

        uint32_t blocks = 32;
        gpu_setup_curand<TFloat> <<<32, 32>>>
            (seed, curand_states, num_generators);
        CudaCheckError();
    
#ifdef _DEBUG
        cudaDeviceSynchronize();
    
        cudaPrintfDisplay(stdout, true);
        cudaPrintfEnd();
        std::cout << "END Kernel Printing ...\n";
#endif    
    }

    // Explicit specialization
    template void gpu_setup_curand_wrapper<float>(uint64_t seed,
                                                  curandState *curand_states,
                                                  uint32_t num_generators);

    template void gpu_setup_curand_wrapper<double>(uint64_t seed,
                                                   curandState *curand_states,
                                                   uint32_t num_generators);

}
