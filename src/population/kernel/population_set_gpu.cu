#include "../population_set_gpu.h"

namespace locusta {

    /// GPU Kernels Shared Memory Pointer.
    extern __shared__ int ga_shared_memory[];

    template <typename TFloat>
    __global__
    void initialize_device_population_set_kernel
    (const uint32_t NUM_ISLES,
     const uint32_t NUM_AGENTS,
     const uint32_t NUM_DIMENSIONS,
     const TFloat * const LOWER_BOUNDS,
     const TFloat * const VAR_RANGES,
     TFloat * const data_array,
     TFloat * const transformed_data_array,
     TFloat * const fitness_array)
    {
        const uint32_t idx =
            threadIdx.x +
            blockIdx.x * NUM_AGENTS;

        for(uint32_t i = 0; i < NUM_DIMENSIONS; ++i)
        {        
            data_array[idx + i * NUM_ISLES * NUM_AGENTS] =
                LOWER_BOUNDS[i] + (VAR_RANGES[i] * transformed_data_array[idx + i * NUM_ISLES * NUM_AGENTS]);
        
            transformed_data_array[idx + i * NUM_ISLES * NUM_AGENTS] =
                (CUDART_NAN_F);
        }
        fitness_array[idx] = -(CUDART_NAN_F);
    }
  
    template <typename TFloat>
    __global__
    void update_records_kernel
    (uint32_t * const highest_idx,
     uint32_t * const lowest_idx,
     TFloat * const highest_fitness,
     TFloat * const lowest_fitness,
     TFloat * const fitness_array)
    {
        // GPU Shared memory
        uint32_t *max_idx_reduction = (uint32_t *) ga_shared_memory;
        uint32_t *min_idx_reduction = (uint32_t *) &max_idx_reduction[blockDim.x];

        TFloat *max_eval_reduction = (TFloat *) &min_idx_reduction[blockDim.x];
        TFloat *min_eval_reduction = (TFloat *) &max_eval_reduction[blockDim.x];

        TFloat a, b;
    
        // Initialize per thread fitness values
        max_eval_reduction[threadIdx.x] = min_eval_reduction[threadIdx.x] =
            fitness_array[blockIdx.x * blockDim.x + threadIdx.x];
        max_idx_reduction[threadIdx.x] = min_idx_reduction[threadIdx.x] =
            threadIdx.x;

        int reduction_idx = 1;
        const int reduction_limit = blockDim.x;

        // Calculate reduction steps
        while(reduction_idx < reduction_limit)
        {
            reduction_idx <<= 1;
        }

        // Parallel reduction
        while(reduction_idx != 0)
        {
            if(threadIdx.x < reduction_idx &&
               threadIdx.x + reduction_idx < reduction_limit)
            {
                a = max_eval_reduction[threadIdx.x];
                b = max_eval_reduction[threadIdx.x + reduction_idx];

                if(b > a)
                {
                    max_eval_reduction[threadIdx.x] = b;
                    max_idx_reduction[threadIdx.x] = max_idx_reduction[threadIdx.x + reduction_idx];
                }

                a = min_eval_reduction[threadIdx.x];
                b = min_eval_reduction[threadIdx.x + reduction_idx];

                if(b < a)
                {
                    min_eval_reduction[threadIdx.x] = b;
                    min_idx_reduction[threadIdx.x] = min_idx_reduction[threadIdx.x + reduction_idx];
                }
            }
            reduction_idx >>= 1;
            __syncthreads();
        }

        // Store parallel reduction results
        if(threadIdx.x == 0)
        {
            highest_fitness[blockIdx.x] = max_eval_reduction[0];
            highest_idx[blockIdx.x] = max_idx_reduction[0];

            lowest_fitness[blockIdx.x] = min_eval_reduction[0];
            lowest_idx[blockIdx.x] = min_idx_reduction[0];
        }
        return;
    }

    template <typename TFloat>
    void initialize_device_population_set_wrapper
    (const uint32_t NUM_ISLES,
     const uint32_t NUM_AGENTS,
     const uint32_t NUM_DIMENSIONS,
     const TFloat * const LOWER_BOUNDS,
     const TFloat * const VAR_RANGES,
     TFloat * const data_array,
     TFloat * const transformed_data_array,
     TFloat * const fitness_array)
    {
        initialize_device_population_set_kernel<<<NUM_ISLES, NUM_AGENTS>>>
            (NUM_ISLES,
             NUM_AGENTS,
             NUM_DIMENSIONS,
             LOWER_BOUNDS,
             VAR_RANGES,
             data_array,
             transformed_data_array,
             fitness_array);
        CudaCheckError();
    }

    template <typename TFloat>
    void update_records_wrapper
    (const uint32_t NUM_ISLES,
     const uint32_t NUM_AGENTS,
     uint32_t * const highest_idx,
     uint32_t * const lowest_idx,
     TFloat * const highest_fitness,
     TFloat * const lowest_fitness,
     TFloat * const fitness_array)
    {
        update_records_kernel
            <<<NUM_ISLES, NUM_AGENTS, 2 * NUM_AGENTS * (sizeof(int) + sizeof(TFloat))>>>
            (highest_idx,
             lowest_idx,
             highest_fitness,
             lowest_fitness,
             fitness_array);
        CudaCheckError();
    }
  
    // Template Specialization (float)
  
    template
    void initialize_device_population_set_wrapper<float>
    (const uint32_t NUM_ISLES,
     const uint32_t NUM_AGENTS,
     const uint32_t NUM_DIMENSIONS,
     const float * const LOWER_BOUNDS,
     const float * const VAR_RANGES,
     float * const data_array,
     float * const transformed_data_array,
     float * const fitness_array);
  
    template
    void update_records_wrapper<float>
    (const uint32_t NUM_ISLES,
     const uint32_t NUM_AGENTS,
     uint32_t * const highest_idx,
     uint32_t * const lowest_idx,
     float * const highest_fitness,
     float * const lowest_fitness,
     float * const fitness_array);

    // Template Specialization (double)
  
    template
    void initialize_device_population_set_wrapper<double>
    (const uint32_t NUM_ISLES,
     const uint32_t NUM_AGENTS,
     const uint32_t NUM_DIMENSIONS,
     const double * const LOWER_BOUNDS,
     const double * const VAR_RANGES,
     double * const data_array,
     double * const transformed_data_array,
     double * const fitness_array);
  
    template
    void update_records_wrapper<double>
    (const uint32_t NUM_ISLES,
     const uint32_t NUM_AGENTS,
     uint32_t * const highest_idx,
     uint32_t * const lowest_idx,
     double * const highest_fitness,
     double * const lowest_fitness,
     double * const fitness_array);

}
