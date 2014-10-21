#include "cuda_common/cuda_helpers.h"
#include "population_set_cuda.hpp"

namespace locusta {

    template <typename TFloat>
    population_set_cuda<TFloat>::population_set_cuda
    (const uint32_t ISLES,
     const uint32_t AGENTS,
     const uint32_t DIMENSIONS)
        : population_set<TFloat>(ISLES, AGENTS, DIMENSIONS)
    {
        // Device Memory Allocation
        CudaSafeCall(cudaMalloc((void **) &_dev_data_array, _TOTAL_GENES * sizeof(TFloat)));
        CudaSafeCall(cudaMalloc((void **) &_dev_transformed_data_array, _TOTAL_GENES * sizeof(TFloat)));
        CudaSafeCall(cudaMalloc((void **) &_dev_fitness_array, _TOTAL_AGENTS * sizeof(TFloat)));
    }

    template <typename TFloat>
    population_set_cuda<TFloat>::~population_set_cuda()
    {
        CudaSafeCall(cudaFree(_dev_data_array));
        CudaSafeCall(cudaFree(_dev_transformed_data_array));
        CudaSafeCall(cudaFree(_dev_fitness_array));
    }

    template <typename TFloat>
    void population_set_cuda<TFloat>::gen_cpy(TFloat * dst_data,
                                              const TFloat * src_data,
                                              size_t elements,
                                              GenomeCopyKind copy_type)
    {
        const size_t bytes_2_copy = elements * sizeof(TFloat);

        TFloat * store_buffer;

        switch(copy_type)
        {
        case GencpyHostToHost:
            memcpy(dst_data, src_data, bytes_2_copy);
            break;
        case GencpyDeviceToDevice:
            CudaSafeCall(cudaMemcpy(dst_data, src_data, bytes_2_copy, cudaMemcpyDeviceToDevice));
            break;
        case GencpyDeviceToHost:
            store_buffer = new TFloat[bytes_2_copy];
            // Copy data from CUDA into temporal buffer.
            CudaSafeCall(cudaMemcpy(store_buffer, src_data, bytes_2_copy, cudaMemcpyDeviceToHost));
            // Rearange genomes into CPU scheme.
            for (uint32_t i = 0; i < _ISLES; ++i)
            {
                for(uint32_t j = 0; j < _AGENTS; ++j)
                {
                    const uint32_t locus_offset = i * _AGENTS + j;
                    for(uint32_t k = 0; k < _DIMENSIONS; ++k)
                    {
                        const uint32_t cpu_idx = i * _AGENTS * _DIMENSIONS + j * _DIMENSIONS + k;
                        const uint32_t cuda_idx = k * _ISLES * _AGENTS + locus_offset;
                        dst_data[cpu_idx] = store_buffer[cuda_idx];
                    }
                }
            }

            delete [] store_buffer;
            break;
        case GencpyHostToDevice:
            store_buffer = new TFloat[bytes_2_copy];
            // Rearange genomes into CUDA scheme in temporal buffer.
            for (uint32_t i = 0; i < _ISLES; ++i)
            {
                for(uint32_t j = 0; j < _AGENTS; ++j)
                {
                    const uint32_t locus_offset = i * _AGENTS + j;
                    for(uint32_t k = 0; k < _DIMENSIONS; ++k)
                    {
                        const uint32_t cpu_idx = i * _AGENTS * _DIMENSIONS + j * _DIMENSIONS + k;
                        const uint32_t cuda_idx = k * _ISLES * _AGENTS + locus_offset;
                        store_buffer[cuda_idx] = src_data[cpu_idx];
                    }
                }
            }
            // Copy rearranged buffer into CUDA
            CudaSafeCall(cudaMemcpy(dst_data, store_buffer, bytes_2_copy, cudaMemcpyHostToDevice));

            delete [] store_buffer;
            break;
        }
    }

    template <typename TFloat>
    void population_set_cuda<TFloat>::swap_data_sets()
    {
        TFloat * swapped_ptr = _dev_data_array;
        _dev_data_array = _dev_transformed_data_array;
        _dev_transformed_data_array = swapped_ptr;
    }

}
