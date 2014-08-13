#include "cuda_common/cuda_helpers.h"

namespace locusta {

    /// Cuda Wrappers Forward Declarations
    template <typename TFloat>
        void initialize_device_population_set_dispatch
        (const uint32_t NUM_ISLES,
         const uint32_t NUM_AGENTS,
         const uint32_t NUM_DIMENSIONS,
         const TFloat * const LOWER_BOUNDS,
         const TFloat * const VAR_RANGES,
         TFloat * const data_array,
         TFloat * const transformed_data_array,
         TFloat * const fitness_array);

    template <typename TFloat>
        void update_records_dispatch
        (const uint32_t NUM_ISLES,
         const uint32_t NUM_AGENTS,
         uint32_t * const highest_idx,
         uint32_t * const lowest_idx,
         TFloat * const highest_fitness,
         TFloat * const lowest_fitness,
         TFloat * const fitness_array);

    template <typename TFloat>
        population_set_gpu<TFloat>::population_set_gpu
        (const uint32_t NUM_ISLES,
         const uint32_t NUM_AGENTS,
         const uint32_t NUM_DIMENSIONS,
         TFloat * upper_bound,
         TFloat * lower_bound)
        : population_set<TFloat>(NUM_ISLES,
                                 NUM_AGENTS,
                                 NUM_DIMENSIONS,
                                 upper_bound,
                                 lower_bound)
    {
        /// Host Memory Allocation
        _var_ranges = new TFloat[NUM_DIMENSIONS];
        _highest_idx = new uint32_t[NUM_ISLES];
        _lowest_idx = new uint32_t[NUM_ISLES];
        _highest_fitness = new TFloat[NUM_ISLES];
        _lowest_fitness = new TFloat[NUM_ISLES];

        /// Device Memory Allocation
        CudaSafeCall(cudaMalloc((void **) &_dev_var_ranges, NUM_DIMENSIONS * sizeof(TFloat)));
        CudaSafeCall(cudaMalloc((void **) &_DEV_UPPER_BOUNDS, NUM_DIMENSIONS * sizeof(TFloat)));
        CudaSafeCall(cudaMalloc((void **) &_DEV_LOWER_BOUNDS, NUM_DIMENSIONS * sizeof(TFloat)));

        CudaSafeCall(cudaMalloc((void **) &_dev_data_array,
                                NUM_DIMENSIONS * NUM_AGENTS * NUM_ISLES * sizeof(TFloat)));
        CudaSafeCall(cudaMalloc((void **) &_dev_transformed_data_array,
                                NUM_DIMENSIONS * NUM_AGENTS * NUM_ISLES * sizeof(TFloat)));

        CudaSafeCall(cudaMalloc((void **) &_dev_fitness_array, NUM_AGENTS * NUM_ISLES * sizeof(TFloat)));

        CudaSafeCall(cudaMalloc((void **) &_dev_highest_idx, NUM_ISLES * sizeof(uint32_t)));
        CudaSafeCall(cudaMalloc((void **) &_dev_lowest_idx, NUM_ISLES * sizeof(uint32_t)));
        CudaSafeCall(cudaMalloc((void **) &_dev_highest_fitness, NUM_ISLES * sizeof(TFloat)));
        CudaSafeCall(cudaMalloc((void **) &_dev_lowest_fitness, NUM_ISLES * sizeof(TFloat)));

        CudaCheckError();
    }

    template <typename TFloat>
        population_set_gpu<TFloat>::~population_set_gpu()
    {
        delete [] _var_ranges;
        delete [] _highest_idx;
        delete [] _lowest_idx;
        delete [] _highest_fitness;
        delete [] _lowest_fitness;

        CudaSafeCall(cudaFree(_dev_var_ranges));
        CudaSafeCall(cudaFree(_DEV_UPPER_BOUNDS));
        CudaSafeCall(cudaFree(_DEV_LOWER_BOUNDS));

        CudaSafeCall(cudaFree(_dev_data_array));
        CudaSafeCall(cudaFree(_dev_transformed_data_array));
        CudaSafeCall(cudaFree(_dev_fitness_array));

        CudaSafeCall(cudaFree(_dev_highest_idx));
        CudaSafeCall(cudaFree(_dev_lowest_idx));
        CudaSafeCall(cudaFree(_dev_highest_fitness));
        CudaSafeCall(cudaFree(_dev_lowest_fitness));
    }


    template <typename TFloat>
        void population_set_gpu<TFloat>::_initialize()
    {
        /// Value Initialization
        for(uint32_t i = 0; i < _NUM_DIMENSIONS; ++i)
        {
            _var_ranges[i] = _UPPER_BOUNDS[i] - _LOWER_BOUNDS[i];
        }

        for(uint32_t j = 0; j < _NUM_ISLES; ++j)
        {
            _highest_idx[j] = 0;
            _lowest_idx[j] = 0;
            _highest_fitness[j] = -std::numeric_limits<TFloat>::infinity();
            _lowest_fitness[j] = std::numeric_limits<TFloat>::infinity();
        }

        CudaSafeCall(cudaMemcpy(_dev_var_ranges,
                                _var_ranges,
                                sizeof(TFloat) * _NUM_DIMENSIONS,
                                cudaMemcpyHostToDevice));

        CudaSafeCall(cudaMemcpy(_DEV_UPPER_BOUNDS,
                                _UPPER_BOUNDS,
                                sizeof(TFloat) * _NUM_DIMENSIONS,
                                cudaMemcpyHostToDevice));

        CudaSafeCall(cudaMemcpy(_DEV_LOWER_BOUNDS,
                                _LOWER_BOUNDS,
                                sizeof(TFloat) * _NUM_DIMENSIONS,
                                cudaMemcpyHostToDevice));

        initialize_device_population_set_dispatch<TFloat>(_NUM_ISLES,
                                                         _NUM_AGENTS,
                                                         _NUM_DIMENSIONS,
                                                         _DEV_LOWER_BOUNDS,
                                                         _dev_var_ranges,
                                                         _dev_data_array,
                                                         _dev_transformed_data_array,
                                                         _dev_fitness_array);
    }

    template <typename TFloat>
        void population_set_gpu<TFloat>::_print_data()
    {
        TFloat * local_data_array = new TFloat[_NUM_ISLES * _NUM_AGENTS * _NUM_DIMENSIONS];
        TFloat * local_fitness_array = new TFloat[_NUM_ISLES * _NUM_AGENTS];

        CudaSafeCall(cudaMemcpy(local_data_array,
                                _dev_data_array,
                                sizeof(TFloat) * _NUM_ISLES * _NUM_AGENTS * _NUM_DIMENSIONS,
                                cudaMemcpyDeviceToHost));

        CudaSafeCall(cudaMemcpy(local_fitness_array,
                                _dev_fitness_array,
                                sizeof(TFloat) * _NUM_ISLES * _NUM_AGENTS,
                                cudaMemcpyDeviceToHost));

        for(uint32_t isle_count = 0; isle_count < _NUM_ISLES; ++isle_count)
        {
            std::cout << "Isle: " << isle_count << std::endl;
            for(uint32_t agent_count = 0; agent_count < _NUM_AGENTS; ++agent_count)
            {
                const uint32_t gene_base_idx =
                    isle_count * _NUM_DIMENSIONS * _NUM_AGENTS +
                    agent_count;

                const uint32_t fitness_idx =
                    isle_count * _NUM_AGENTS + agent_count;

                std::cout << "\tAgent: " << agent_count <<
                    ", F: "<< local_fitness_array[fitness_idx] <<
                    "\n\t";

                for(uint32_t k = 0; k < _NUM_DIMENSIONS; ++k)
                {
                    std::cout << local_data_array[gene_base_idx + k * _NUM_AGENTS] << ", ";
                }
                std::cout << std::endl;
            }
        }

        delete local_data_array;
        delete local_fitness_array;
    }

    template <typename TFloat>
        void population_set_gpu<TFloat>::_swap_data_sets()
    {
        TFloat * const _tmp_data_pointer = _dev_data_array;
        _dev_data_array = _dev_transformed_data_array;
        _dev_transformed_data_array = _tmp_data_pointer;
    }

    template <typename TFloat>
        void population_set_gpu<TFloat>::_update_records()
    {
        update_records_dispatch(_NUM_ISLES,
                               _NUM_AGENTS,
                               _dev_highest_idx,
                               _dev_lowest_idx,
                               _dev_highest_fitness,
                               _dev_lowest_fitness,
                               _dev_fitness_array);

        /// Copy results to host memory
        CudaSafeCall(cudaMemcpy(_highest_idx,
                                _dev_highest_idx,
                                sizeof(uint32_t) * _NUM_ISLES,
                                cudaMemcpyDeviceToHost));

        CudaSafeCall(cudaMemcpy(_lowest_idx,
                                _dev_lowest_idx,
                                sizeof(uint32_t) * _NUM_ISLES,
                                cudaMemcpyDeviceToHost));

        CudaSafeCall(cudaMemcpy(_highest_fitness,
                                _dev_highest_fitness,
                                sizeof(TFloat) * _NUM_ISLES,
                                cudaMemcpyDeviceToHost));

        CudaSafeCall(cudaMemcpy(_lowest_fitness,
                                _dev_lowest_fitness,
                                sizeof(TFloat) * _NUM_ISLES,
                                cudaMemcpyDeviceToHost));
    }

    template <typename TFloat>
        TFloat * population_set_gpu<TFloat>::_get_dev_data_array()
    {
        return _dev_data_array;
    }

    template <typename TFloat>
        TFloat * population_set_gpu<TFloat>::_get_dev_transformed_data_array()
    {
        return _dev_transformed_data_array;
    }

    template <typename TFloat>
        TFloat * population_set_gpu<TFloat>::_get_dev_fitness_array()
    {
        return _dev_fitness_array;
    }

    template <typename TFloat>
        const TFloat * population_set_gpu<TFloat>::_get_dev_upper_bounds()
    {
        return _DEV_UPPER_BOUNDS;
    }

    template <typename TFloat>
        const TFloat * population_set_gpu<TFloat>::_get_dev_lower_bounds()
    {
        return _DEV_LOWER_BOUNDS;
    }

    template <typename TFloat>
        void population_set_gpu<TFloat>::_copy_dev_data_into_host(TFloat * const output_data_array)
    {
        CudaSafeCall(cudaMemcpy(output_data_array,
                                _dev_data_array,
                                sizeof(TFloat) * _NUM_ISLES * _NUM_AGENTS * _NUM_DIMENSIONS,
                                cudaMemcpyDeviceToHost));

        return;
    }

    template <typename TFloat>
        void population_set_gpu<TFloat>::_copy_host_data_into_dev(const TFloat * const input_data_array)
    {
        CudaSafeCall(cudaMemcpy(_dev_data_array,
                                input_data_array,
                                sizeof(TFloat) * _NUM_ISLES * _NUM_AGENTS * _NUM_DIMENSIONS,
                                cudaMemcpyHostToDevice));

        return;
    }

    template <typename TFloat>
        void population_set_gpu<TFloat>::_copy_dev_fitness_into_host(TFloat * const output_fitness_array)
    {
        CudaSafeCall(cudaMemcpy(output_fitness_array,
                                _dev_fitness_array,
                                sizeof(TFloat) * _NUM_ISLES * _NUM_AGENTS,
                                cudaMemcpyDeviceToHost));
        return;
    }

    template <typename TFloat>
        void population_set_gpu<TFloat>::_copy_host_fitness_into_dev(const TFloat * const input_fitness_array)
    {
        CudaSafeCall(cudaMemcpy(_dev_fitness_array,
                                input_fitness_array,
                                sizeof(TFloat) * _NUM_ISLES * _NUM_AGENTS,
                                cudaMemcpyHostToDevice));
        return;
    }

}
