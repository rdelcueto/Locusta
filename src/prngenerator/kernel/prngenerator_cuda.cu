#include <iostream>

#include "../prngenerator_cuda.hpp"

namespace locusta {

/**
 * @brief CUDA kernel for setting up cuRAND states.
 *
 * This kernel initializes an array of cuRAND states on the GPU.
 *
 * @param seed Seed value for the PRNG engines.
 * @param curand_states Pointer to the array of cuRAND states.
 * @param num_engines Number of PRNG engines to use.
 */
template<typename TFloat>
__global__ void
gpu_setup_curand(uint64_t seed,
                 curandState* curand_states,
                 uint32_t num_engines)
{
  uint64_t id =
    blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

  while (id < num_engines) {
    curand_init(id + seed, 0, 0, &curand_states[id]);
    id += blockDim.x * gridDim.x + blockDim.y * blockDim.y;
  }
}

/**
 * @brief Dispatch function for setting up cuRAND states on the GPU.
 *
 * This function dispatches the setup of cuRAND states using the
 * corresponding engines.
 *
 * @param seed Seed value for the PRNG engines.
 * @param curand_states Pointer to the array of cuRAND states.
 * @param num_engines Number of PRNG engines to use.
 */
template<typename TFloat>
void
gpu_setup_curand_dispatch(uint64_t seed,
                          curandState* curand_states,
                          uint32_t num_engines)
{
  gpu_setup_curand<TFloat><<<32, 32>>>(seed, curand_states, num_engines);
  CudaCheckError();
}

template<typename TFloat>
__global__ void
gpu_cast_float_to_tfloat(const float* __restrict__ in, TFloat* __restrict__ out, uint32_t n)
{
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = static_cast<TFloat>(in[idx]);
  }
}

template<typename TFloat>
void
gpu_cast_float_to_tfloat_dispatch(const float* in, TFloat* out, uint32_t n)
{
  uint32_t threads = 256;
  uint32_t blocks = (n + threads - 1) / threads;
  gpu_cast_float_to_tfloat<TFloat><<<blocks, threads>>>(in, out, n);
  CudaCheckError();
}

// Explicit specializations
template void gpu_setup_curand_dispatch<float>(uint64_t, curandState*, uint32_t);
template void gpu_setup_curand_dispatch<double>(uint64_t, curandState*, uint32_t);
template void gpu_setup_curand_dispatch<__half>(uint64_t, curandState*, uint32_t);
template void gpu_setup_curand_dispatch<__nv_bfloat16>(uint64_t, curandState*, uint32_t);
template void gpu_setup_curand_dispatch<__nv_fp8_e4m3>(uint64_t, curandState*, uint32_t);
template void gpu_setup_curand_dispatch<__nv_fp8_e5m2>(uint64_t, curandState*, uint32_t);

template void gpu_cast_float_to_tfloat_dispatch<float>(const float*, float*, uint32_t);
template void gpu_cast_float_to_tfloat_dispatch<double>(const float*, double*, uint32_t);
template void gpu_cast_float_to_tfloat_dispatch<__half>(const float*, __half*, uint32_t);
template void gpu_cast_float_to_tfloat_dispatch<__nv_bfloat16>(const float*, __nv_bfloat16*, uint32_t);
template void gpu_cast_float_to_tfloat_dispatch<__nv_fp8_e4m3>(const float*, __nv_fp8_e4m3*, uint32_t);
template void gpu_cast_float_to_tfloat_dispatch<__nv_fp8_e5m2>(const float*, __nv_fp8_e5m2*, uint32_t);

}
