namespace locusta {

template <typename TFloat>
void initialize_vector_dispatch(const uint32_t ISLES, const uint32_t AGENTS,
                                const uint32_t DIMENSIONS,
                                const TFloat* LOWER_BOUNDS,
                                const TFloat* VAR_RANGES, const TFloat* tmp_vec,
                                TFloat* dst_vec);

template <typename TFloat>
void crop_vector_dispatch(const uint32_t ISLES, const uint32_t AGENTS,
                          const uint32_t DIMENSIONS, const TFloat* UPPER_BOUNDS,
                          const TFloat* LOWER_BOUNDS, TFloat* vec);

template <typename TFloat>
void update_records_dispatch(const uint32_t ISLES, const uint32_t AGENTS,
                             const uint32_t DIMENSIONS,
                             const TFloat* data_array,
                             const TFloat* fitness_array,
                             TFloat* max_agent_genome, TFloat* min_agent_genome,
                             TFloat* max_agent_fitness,
                             TFloat* min_agent_fitness);

template <typename TFloat>
evolutionary_solver_cuda<TFloat>::evolutionary_solver_cuda(
  population_set_cuda<TFloat>* population, evaluator_cuda<TFloat>* evaluator,
  prngenerator_cuda<TFloat>* prn_generator, uint32_t generation_target,
  TFloat* upper_bounds, TFloat* lower_bounds)
  : evolutionary_solver<TFloat>(population, evaluator, prn_generator,
                                generation_target, upper_bounds, lower_bounds)
  , _dev_population(static_cast<population_set_cuda<TFloat>*>(_population))
  , _dev_evaluator(static_cast<evaluator_cuda<TFloat>*>(_evaluator))
  , _dev_bulk_prn_generator(
      static_cast<prngenerator_cuda<TFloat>*>(prn_generator))
{
  // Allocate Device Memory
  CudaSafeCall(
    cudaMalloc((void**)&(_DEV_UPPER_BOUNDS), _DIMENSIONS * sizeof(TFloat)));
  CudaSafeCall(
    cudaMalloc((void**)&(_DEV_LOWER_BOUNDS), _DIMENSIONS * sizeof(TFloat)));
  CudaSafeCall(
    cudaMalloc((void**)&(_DEV_VAR_RANGES), _DIMENSIONS * sizeof(TFloat)));

  CudaSafeCall(cudaMalloc((void**)&(_dev_max_agent_genome),
                          _ISLES * _DIMENSIONS * sizeof(TFloat)));
  CudaSafeCall(
    cudaMalloc((void**)&(_dev_max_agent_fitness), _ISLES * sizeof(TFloat)));
  CudaSafeCall(
    cudaMalloc((void**)&(_dev_max_agent_idx), _ISLES * sizeof(uint32_t)));

  CudaSafeCall(cudaMalloc((void**)&(_dev_min_agent_genome),
                          _ISLES * _DIMENSIONS * sizeof(TFloat)));
  CudaSafeCall(
    cudaMalloc((void**)&(_dev_min_agent_fitness), _ISLES * sizeof(TFloat)));
  CudaSafeCall(
    cudaMalloc((void**)&(_dev_min_agent_idx), _ISLES * sizeof(uint32_t)));

  // Copy values into device
  CudaSafeCall(cudaMemcpy(_DEV_UPPER_BOUNDS, _UPPER_BOUNDS,
                          _DIMENSIONS * sizeof(TFloat),
                          cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(_DEV_LOWER_BOUNDS, _LOWER_BOUNDS,
                          _DIMENSIONS * sizeof(TFloat),
                          cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(_DEV_VAR_RANGES, _VAR_RANGES,
                          _DIMENSIONS * sizeof(TFloat),
                          cudaMemcpyHostToDevice));
}

template <typename TFloat>
evolutionary_solver_cuda<TFloat>::~evolutionary_solver_cuda()
{
  // Free Device memory
  CudaSafeCall(cudaFree(_DEV_UPPER_BOUNDS));
  CudaSafeCall(cudaFree(_DEV_LOWER_BOUNDS));
  CudaSafeCall(cudaFree(_DEV_VAR_RANGES));

  CudaSafeCall(cudaFree(_dev_max_agent_genome));
  CudaSafeCall(cudaFree(_dev_max_agent_fitness));
  CudaSafeCall(cudaFree(_dev_max_agent_idx));

  CudaSafeCall(cudaFree(_dev_min_agent_genome));
  CudaSafeCall(cudaFree(_dev_min_agent_fitness));
  CudaSafeCall(cudaFree(_dev_min_agent_idx));
}

template <typename TFloat>
void
evolutionary_solver_cuda<TFloat>::setup_solver()
{
  // Initialize Population
  if (!_population->_f_initialized) {
    initialize_vector(_dev_population->_dev_data_array);
    _population->_f_initialized = true;
  }

  // Initialize solver records
  const TFloat* data_array = const_cast<TFloat*>(_population->_data_array);
  for (uint32_t i = 0; i < _ISLES; ++i) {
    // Initialize best genomes
    _max_agent_idx[i] = 0;
    _min_agent_idx[i] = 0;

    _max_agent_fitness[i] = -std::numeric_limits<TFloat>::infinity();
    _min_agent_fitness[i] = std::numeric_limits<TFloat>::infinity();

    for (uint32_t k = 0; k < _DIMENSIONS; ++k) {
      const uint32_t data_idx = i * _AGENTS * _DIMENSIONS + 0 * _DIMENSIONS + k;
      _max_agent_genome[i + k * _ISLES] = _min_agent_genome[i + k * _ISLES] = 0;
      // TODO: Define correct initialize value.
      // std::numeric_limits<TFloat>::quiet_NaN();
    }
  }

  CudaSafeCall(cudaMemcpy(_dev_max_agent_genome, _max_agent_genome,
                          _ISLES * _DIMENSIONS * sizeof(TFloat),
                          cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(_dev_max_agent_fitness, _max_agent_fitness,
                          _ISLES * sizeof(TFloat), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(_dev_max_agent_idx, _max_agent_idx,
                          _ISLES * sizeof(uint32_t), cudaMemcpyHostToDevice));

  CudaSafeCall(cudaMemcpy(_dev_min_agent_genome, _min_agent_genome,
                          _ISLES * _DIMENSIONS * sizeof(TFloat),
                          cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(_dev_min_agent_fitness, _min_agent_fitness,
                          _ISLES * sizeof(TFloat), cudaMemcpyHostToDevice));
  CudaSafeCall(cudaMemcpy(_dev_min_agent_idx, _min_agent_idx,
                          _ISLES * sizeof(uint32_t), cudaMemcpyHostToDevice));

  // Initialize fitness, evaluating initialization data.
  evaluate_genomes();
  // Update solver records.
  update_records();
  // Initialize random numbers.
  regenerate_prns();
}

template <typename TFloat>
void
evolutionary_solver_cuda<TFloat>::evaluate_genomes()
{
  _dev_evaluator->evaluate(this);
}

template <typename TFloat>
void
evolutionary_solver_cuda<TFloat>::update_records()
{
  const TFloat* data_array = _dev_population->_dev_data_array;
  const TFloat* fitness_array = _dev_population->_dev_fitness_array;

  update_records_dispatch(_ISLES, _AGENTS, _DIMENSIONS, data_array,
                          fitness_array, _dev_max_agent_genome,
                          _dev_min_agent_genome, _dev_max_agent_fitness,
                          _dev_min_agent_fitness);
}

template <typename TFloat>
void
evolutionary_solver_cuda<TFloat>::regenerate_prns()
{
  _bulk_prn_generator->_generate(_bulk_size, _dev_bulk_prns);
}

template <typename TFloat>
void
evolutionary_solver_cuda<TFloat>::crop_vector(TFloat* vec)
{
  crop_vector_dispatch(_ISLES, _AGENTS, _DIMENSIONS, _DEV_UPPER_BOUNDS,
                       _DEV_LOWER_BOUNDS, vec);
}

template <typename TFloat>
void
evolutionary_solver_cuda<TFloat>::initialize_vector(TFloat* dst_vec)
{
  const uint32_t TOTAL_GENES = _population->_TOTAL_GENES;

  TFloat* tmp_vec;
  CudaSafeCall(cudaMalloc((void**)&(tmp_vec), TOTAL_GENES * sizeof(TFloat)));

  // Initialize vector data, within given bounds.
  _dev_bulk_prn_generator->_generate(TOTAL_GENES, tmp_vec);

  initialize_vector_dispatch(_ISLES, _AGENTS, _DIMENSIONS, _DEV_LOWER_BOUNDS,
                             _DEV_VAR_RANGES, tmp_vec, dst_vec);

  CudaSafeCall(cudaFree(tmp_vec));
}

template <typename TFloat>
void
evolutionary_solver_cuda<TFloat>::print_transformation_diff()
{
  // Copy population into host
  const uint32_t genes = _dev_population->_TOTAL_GENES;
  const TFloat* const device_data = _dev_population->_dev_data_array;
  TFloat* const host_data = _dev_population->_data_array;

  _dev_population->gen_cpy(host_data, device_data, genes, GencpyDeviceToHost);

  const TFloat* const transformed_data =
    _dev_population->_dev_transformed_data_array;
  TFloat* const host_transformed_data =
    _dev_population->_transformed_data_array;

  _dev_population->gen_cpy(host_transformed_data, transformed_data, genes,
                           GencpyDeviceToHost);

  for (uint32_t i = 0; i < _ISLES; i++) {
    for (uint32_t j = 0; j < _AGENTS; j++) {
      std::cout << "[" << i << ", " << j << "] : ";
      TFloat* a = host_data + i * _AGENTS * _DIMENSIONS + j * _DIMENSIONS;
      TFloat* b =
        host_transformed_data + i * _AGENTS * _DIMENSIONS + j * _DIMENSIONS;
      for (uint32_t k = 0; k < _DIMENSIONS; k++) {
        std::cout << (a[k] - b[k]);
        if (k == _DIMENSIONS - 1) {
          std::cout << "]\n";
        } else {
          std::cout << ", ";
        }
      }
    }
  }
}

template <typename TFloat>
void
evolutionary_solver_cuda<TFloat>::print_population()
{
  // Copy population into host
  const uint32_t genes = _dev_population->_TOTAL_GENES;
  const TFloat* const device_data = _dev_population->_dev_data_array;
  TFloat* const host_data = _dev_population->_data_array;

  _dev_population->gen_cpy(host_data, device_data, genes, GencpyDeviceToHost);

  // Copy fitness into host
  const uint32_t agents = _dev_population->_TOTAL_AGENTS;
  const TFloat* const device_fitness = _dev_population->_dev_fitness_array;
  TFloat* const host_fitness_copy = _dev_population->_fitness_array;

  CudaSafeCall(cudaMemcpy(host_fitness_copy, device_fitness,
                          agents * sizeof(TFloat), cudaMemcpyDeviceToHost));

  TFloat* fitness = _population->_fitness_array;
  TFloat* genomes = _population->_data_array;

  for (uint32_t i = 0; i < _ISLES; i++) {
    for (uint32_t j = 0; j < _AGENTS; j++) {
      std::cout << fitness[i * _AGENTS + j] << ", [" << i << ", " << j
                << "] : ";
      TFloat* curr_genome =
        genomes + i * _AGENTS * _DIMENSIONS + j * _DIMENSIONS;
      for (uint32_t k = 0; k < _DIMENSIONS; k++) {
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

template <typename TFloat>
void
evolutionary_solver_cuda<TFloat>::print_solutions()
{
  CudaSafeCall(cudaMemcpy(_max_agent_fitness, _dev_max_agent_fitness,
                          _ISLES * sizeof(TFloat), cudaMemcpyDeviceToHost));

  CudaSafeCall(cudaMemcpy(_max_agent_genome, _dev_max_agent_genome,
                          _ISLES * _DIMENSIONS * sizeof(TFloat),
                          cudaMemcpyDeviceToHost));

  std::cout << "Solutions @ " << (_generation_count) + 1 << " / "
            << _generation_target << std::endl;
  for (uint32_t i = 0; i < _ISLES; i++) {
    std::cout << _max_agent_fitness[i] << " : [";
    for (uint32_t k = 0; k < _DIMENSIONS; k++) {
      std::cout << _max_agent_genome[i + _ISLES * k];
      if (k == _DIMENSIONS - 1) {
        std::cout << "]\n";
      } else {
        std::cout << ", ";
      }
    }
  }
}

} // namespace locusta
