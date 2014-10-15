namespace locusta {

    template<typename TFloat>
    evolutionary_solver_cuda<TFloat>::evolutionary_solver_cuda(population_set_cuda<TFloat> * population,
                                                               evaluator_cuda<TFloat> * evaluator,
                                                               prngenerator_cuda<TFloat> * prn_generator,
                                                               uint32_t generation_target,
                                                               TFloat * upper_bounds,
                                                               TFloat * lower_bounds)
    : evolutionary_solver<TFloat>(population,
                                  evaluator,
                                  prn_generator,
                                  generation_target,
                                  upper_bounds,
                                  lower_bounds),
        _dev_population(static_cast<population_set_cuda<TFloat> *>(_population)),
        _dev_evaluator(static_cast<evaluator_cuda<TFloat> *>(_evaluator)),
        _dev_bulk_prn_generator(static_cast<prngenerator_cuda<TFloat> *>(prn_generator))
    {
        // Allocate Device Memory
        CudaSafeCall(cudaMalloc((void **) &(_DEV_UPPER_BOUNDS), _DIMENSIONS * sizeof(TFloat)));
        CudaSafeCall(cudaMalloc((void **) &(_DEV_LOWER_BOUNDS), _DIMENSIONS * sizeof(TFloat)));
        CudaSafeCall(cudaMalloc((void **) &(_DEV_VAR_RANGES), _DIMENSIONS * sizeof(TFloat)));

        CudaSafeCall(cudaMalloc((void **) &(_dev_best_genome), _ISLES * _DIMENSIONS * sizeof(TFloat)));
        CudaSafeCall(cudaMalloc((void **) &(_dev_best_genome_fitness), _ISLES * sizeof(TFloat)));

        CudaCheckError();

        // Copy values into device
        CudaSafeCall(cudaMemcpy(_DEV_UPPER_BOUNDS, _UPPER_BOUNDS, _DIMENSIONS * sizeof(TFloat), cudaMemcpyHostToDevice));
        CudaSafeCall(cudaMemcpy(_DEV_LOWER_BOUNDS, _LOWER_BOUNDS, _DIMENSIONS * sizeof(TFloat), cudaMemcpyHostToDevice));
        CudaSafeCall(cudaMemcpy(_DEV_VAR_RANGES, _VAR_RANGES, _DIMENSIONS * sizeof(TFloat), cudaMemcpyHostToDevice));

        CudaCheckError();
    }

    template<typename TFloat>
    evolutionary_solver_cuda<TFloat>::~evolutionary_solver_cuda()
    {
        // Free Device memory
        CudaSafeCall(cudaFree(_DEV_UPPER_BOUNDS));
        CudaSafeCall(cudaFree(_DEV_LOWER_BOUNDS));
        CudaSafeCall(cudaFree(_DEV_VAR_RANGES));

        CudaSafeCall(cudaFree(_dev_best_genome));
        CudaSafeCall(cudaFree(_dev_best_genome_fitness));

        CudaCheckError();
    }

    template<typename TFloat>
    void evolutionary_solver_cuda<TFloat>::setup_solver()
    {
        // Initialize Population
        if( !_population->_f_initialized ) {
            initialize_vector(_dev_population->_dev_data_array, _dev_population->_dev_transformed_data_array);
            _population->_f_initialized = true;
        }

        if( _f_initialized ) {
            teardown_solver();
        }

        // Initialize solver records
        const TFloat * data_array = const_cast<TFloat *>(_population->_data_array);
        for(uint32_t i = 0; i < _ISLES; ++i) {
            // Initialize best genomes
            for(uint32_t k = 0; k < _DIMENSIONS; ++k) {
                const uint32_t data_idx =
                    i * _AGENTS * _DIMENSIONS +
                    0 * _DIMENSIONS +
                    k;

                _best_genome[k] = -std::numeric_limits<TFloat>::infinity();
            }
            _best_genome_fitness[i] = -std::numeric_limits<TFloat>::infinity();
        }

        // Copy initialized values into device.
        CudaSafeCall(cudaMemcpy(_dev_best_genome, _best_genome, _ISLES * _DIMENSIONS * sizeof(TFloat), cudaMemcpyHostToDevice));
        CudaSafeCall(cudaMemcpy(_dev_best_genome_fitness, _dev_best_genome_fitness, _ISLES * sizeof(TFloat), cudaMemcpyHostToDevice));

        // Initialize fitness, evaluating initialization data.
        evaluate_genomes();

        // Update solver records
        update_records();
    }


    template<typename TFloat>
    void evolutionary_solver_cuda<TFloat>::advance()
    {
        transform();
        crop_vector(_population->_transformed_data_array);

        _population->swap_data_sets();

        evaluate_genomes();
        update_records();
        regenerate_prnumbers();

        _generation_count++;
    }

    template<typename TFloat>
    void evolutionary_solver_cuda<TFloat>::run()
    {
        do {
            print_solutions();
            advance();
        } while(_generation_count % _generation_target != 0);
    }

    template<typename TFloat>
    void evolutionary_solver_cuda<TFloat>::evaluate_genomes()
    {
        _evaluator->evaluate(this);
    }

    template<typename TFloat>
    void evolutionary_solver_cuda<TFloat>::update_records()
    {

        // TODO: CUDA update_records_kernel dispatch.
    }

    template<typename TFloat>
    void evolutionary_solver_cuda<TFloat>::regenerate_prnumbers()
    {
        _bulk_prn_generator->_generate(_bulk_size, _dev_bulk_prnumbers);
    }

    template<typename TFloat>
    void evolutionary_solver_cuda<TFloat>::crop_vector(TFloat * vec)
    {
        // TODO: CUDA crop_vector_kernel dispatch.
    }

    template<typename TFloat>
    void evolutionary_solver_cuda<TFloat>::initialize_vector(TFloat * dst_vec, TFloat * tmp_vec)
    {
        // TODO: initialize_vector_kernel dispatch
    }

    template<typename TFloat>
    void evolutionary_solver_cuda<TFloat>::print_population()
    {
        // TODO: Copy & rmap device genomes into host memory

        TFloat * fitness = _population->_fitness_array;
        TFloat * genomes = _population->_data_array;

        for(uint32_t i = 0; i < _ISLES; i++) {
            for(uint32_t j = 0; j < _AGENTS; j++) {
                std::cout << fitness[i * _AGENTS + j] << ", [" << i << ", " << j << "] : ";
                TFloat * curr_genome = genomes + i * _AGENTS * _DIMENSIONS + j * _DIMENSIONS;
                for(uint32_t k = 0; k < _DIMENSIONS; k++) {
                    std::cout << curr_genome[k];
                    if (k == _DIMENSIONS - 1) {
                        std::cout << "]\n";
                    } else {
                        std::cout << ", ";
                    }
                }

            }
        }
    }

    template<typename TFloat>
    void evolutionary_solver_cuda<TFloat>::print_solutions()
    {
        // TODO: Copy & rmap device genomes into host memory

        std::cout << "Solutions @ " << (_generation_count)+1 << " / " << _generation_target << std::endl;
        for(uint32_t i = 0; i < _ISLES; i++) {
            std::cout << _best_genome_fitness[i] << " : [";
            for(uint32_t k = 0; k < _DIMENSIONS; k++) {
                std::cout << _best_genome[i*_DIMENSIONS + k];
                if (k == _DIMENSIONS - 1) {
                    std::cout << "]\n";
                } else {
                    std::cout << ", ";
                }
            }
        }
    }

} // namespace locusta
