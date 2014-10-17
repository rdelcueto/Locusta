#include "cuda_common/cuPrintf.cu"
#include <iostream>

#include "../prngenerator_cuda.hpp"

namespace locusta {

    template<typename TFloat>
    __global__ void
    gpu_setup_curand(uint64_t seed,
                     curandState *curand_states,
                     uint32_t num_engines)
    {
        uint64_t id =
            blockIdx.y * gridDim.x * blockDim.x +
            blockIdx.x * blockDim.x +
            threadIdx.x;

        while(id < num_engines)
        {
            curand_init(id + seed, 0, 0, &curand_states[id]);
            id += blockDim.x * gridDim.x + blockDim.y * blockDim.y;
        }
    }

    template<typename TFloat>
    void gpu_setup_curand_dispatch(uint64_t seed,
                                   curandState *curand_states,
                                   uint32_t num_engines)
    {
        gpu_setup_curand<TFloat>
                     <<<32, 32>>>
                     (seed, curand_states, num_engines);
        CudaCheckError();
    }

    // Explicit specialization
    template void gpu_setup_curand_dispatch<float>(uint64_t seed,
                                                   curandState *curand_states,
                                                   uint32_t num_engines);

    template void gpu_setup_curand_dispatch<double>(uint64_t seed,
                                                    curandState *curand_states,
                                                    uint32_t num_engines);

}
