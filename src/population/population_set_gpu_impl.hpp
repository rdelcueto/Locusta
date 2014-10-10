#include "cuda_common/cuda_helpers.h"

namespace locusta {

    enum class GenomeCopyKind { GencpyHostToHost, GencpyHostToDevice, GencpyDeviceToHost, GencpyDeviceToDevice };

    template <typename TFloat>
    population_set_gpu<TFloat>::population_set_gpu
    (const uint32_t ISLES,
     const uint32_t AGENTS,
     const uint32_t DIMENSIONS)
        : population_set<TFloat>(ISLES, AGENTS, DIMENSIONS)
    {
        // Device Memory Allocation
        CudaSafeCall(cudaMalloc((void **) &_dev_data_array, _TOTAL_GENES * sizeof(TFloat)));
        CudaSafeCall(cudaMalloc((void **) &_dev_transformed_data_array, _TOTAL_GENES * sizeof(TFloat)));
        CudaSafeCall(cudaMalloc((void **) &_dev_fitness_array, _TOTAL_AGENTS * sizeof(TFloat)));

        CudaCheckError();
    }

    template <typename TFloat>
    population_set_gpu<TFloat>::~population_set_gpu()
    {
        CudaSafeCall(cudaFree(_dev_data_array));
        CudaSafeCall(cudaFree(_dev_transformed_data_array));
        CudaSafeCall(cudaFree(_dev_fitness_array));

        CudaCheckError();
    }

    template <typename TFloat>
    void population_set_gpu<TFloat>::gen_cpy(TFloat * dst_data,
                                             const TFloat * src_data,
                                             size_t genes,
                                             GenomeCopyKind copy_type)
    {
        const size_t bytes_2_copy = genes * sizeof(TFloat);

        TFloat * store_buffer;

        switch(copy_type)
        {
        case GenomeCopyKind::GencpyHostToHost:
            memcpy(dst_data, src_data, bytes_2_copy);
            break;
        case GenomeCopyKind::GencpyDeviceToDevice:
            cudaMemcpy(dst_data, src_data, bytes_2_copy, cudaMemcpyDeviceToDevice);
            break;
        case GenomeCopyKind::GencpyDeviceToHost:
            store_buffer = new TFloat[bytes_2_copy];
            // Copy data from GPU into temporal buffer.
            cudaMemcpy(store_buffer, src_data, bytes_2_copy, cudaMemcpyDeviceToHost);
            // Rearange genomes into CPU scheme.
            for (uint32_t i = 0; i < _ISLES; ++i)
            {
                for(uint32_t j = 0; j < _AGENTS; ++j)
                {
                    for(uint32_t k = 0; k < _DIMENSIONS; ++k)
                    {
                        const uint32_t cpu_idx = i * _AGENTS * _DIMENSIONS + j * _DIMENSIONS + k;
                        const uint32_t locus_offset = k * _ISLES * _AGENTS;
                        const uint32_t gpu_idx = locus_offset + i * _AGENTS + j;
                        dst_data[cpu_idx] = store_buffer[gpu_idx];
                    }
                }
            }

            delete [] store_buffer;
            break;
        case GenomeCopyKind::GencpyHostToDevice:
            store_buffer = new TFloat[bytes_2_copy];
            // Rearange genomes into GPU scheme in temporal buffer.
            for (uint32_t i = 0; i < _ISLES; ++i)
            {
                for(uint32_t j = 0; j < _AGENTS; ++j)
                {
                    for(uint32_t k = 0; k < _DIMENSIONS; ++k)
                    {
                        const uint32_t cpu_idx = i * _AGENTS * _DIMENSIONS + j * _DIMENSIONS + k;
                        const uint32_t locus_offset = k * _ISLES * _AGENTS;
                        const uint32_t gpu_idx = locus_offset + i * _AGENTS + j;
                        store_buffer[gpu_idx] = src_data[cpu_idx];
                    }
                }
            }
            // Copy rearranged buffer into GPU
            cudaMemcpy(dst_data, store_buffer, bytes_2_copy, cudaMemcpyHostToDevice);
            delete [] store_buffer;
            break;
        }
    }

    template <typename TFloat>
    void population_set_gpu<TFloat>::swap_dev_data_sets()
    {
        TFloat * const swapped_ptr = _dev_data_array;
        _dev_data_array = _dev_transformed_data_array;
        _dev_transformed_data_array = swapped_ptr;
    }

    template <typename TFloat>
    void population_set_gpu<TFloat>::swap_data_sets()
    {
        swap_dev_data_sets();
    }

}
