#include "cuda_common/cuda_helpers.h"

namespace locusta {

  template <typename TFloat>
  ga_solver_gpu<TFloat>::ga_solver_gpu (population_set_gpu<TFloat> * population,
                                        evaluator_gpu<TFloat> * evaluator,
                                        prngenerator_gpu<TFloat> * prn_generator)
    : ga_solver<TFloat>(population,
                        evaluator,
                        prn_generator)
  {
    _dev_population = static_cast<population_set_gpu<TFloat>*>(_population);
  }

  template <typename TFloat>
  ga_solver_gpu<TFloat>::~ga_solver_gpu()
  {
  }

  template <typename TFloat>
  void ga_solver_gpu<TFloat>::_print_solver_config()
  {
    std::cout << "\nGenetic Algorithm Solver (GPU)"
              << "\nConfiguration:"
              << "\n\t * Isles / Agents / Dimensions: "
              << _dev_population->_NUM_ISLES
              << " / " <<  _dev_population->_NUM_AGENTS
              << " / " << _dev_population->_NUM_DIMENSIONS
              << "\n\t * Migration"
              << "\n\t\t Step Size: " << _migration_step
              << "\n\t\t M. Size: " << _migration_size
              << "\n\t\t M. Selection Size: " << _migration_selection_size
              << "\n\t * Selection:"
              << "\n\t\t S. Size: " << _selection_size
              << "\n\t\t S. Stochastic Bias: " << _selection_p
              << "\n\t * Breeding"
              << "\n\t\t Crossover Rate: " << _crossover_rate
              << "\n\t\t Mutation Rate: " << _mutation_rate
              << "\n\t * Domain Range Extension Percent: " << (_range_extension_p) * 10 << " %"
              << "\n\n ";
  }

  template <typename TFloat>
  void ga_solver_gpu<TFloat>::_print_solver_elite()
  {
    const uint32_t NUM_DIMENSIONS = _dev_population->_NUM_DIMENSIONS;
    const uint32_t NUM_ISLES = _dev_population->_NUM_ISLES;

    std::cout << "Elite Genomes:" << std::endl;

    for(uint32_t i = 0; i < NUM_ISLES; ++i)
      {
        std::cout << "Fitness: " << _elite_fitness[i] << std::endl;
        std::cout << "Genome:\n";
        const uint32_t offset = i * NUM_DIMENSIONS;
        for(uint32_t k = 0; k < NUM_DIMENSIONS; ++k)
          {
            std::cout << _elite_genomes[offset + k] << ", ";
          }
        std::cout << std::endl;
      }
    std::cout << "\n\n" << std::endl;
  }

  template <typename TFloat>
  void ga_solver_gpu<TFloat>::_print_solver_solution()
  {
    const uint32_t NUM_ISLES = _dev_population->_NUM_ISLES;
    const uint32_t NUM_DIMENSIONS = _dev_population->_NUM_DIMENSIONS;

    std::cout << "Best Genome:" << std::endl;

    uint32_t best_isle_idx = 0;
    uint32_t best_candidate_idx = 0;
    TFloat best_candidate_fitness = _elite_fitness[0];

    for(uint32_t i = 1; i < NUM_ISLES; ++i)
      {
        if(_elite_fitness[i] > best_candidate_fitness)
          {
            best_isle_idx = i;
            best_candidate_fitness = _elite_fitness[i];
            best_candidate_idx = i;
          }
      }

    std::cout << "Isle: " << best_isle_idx << std::endl;
    std::cout << "Fitness: " << best_candidate_fitness << std::endl;
    std::cout << "Genome:\n";
    const uint32_t offset = best_candidate_idx * NUM_DIMENSIONS;
    for(uint32_t k = 0; k < NUM_DIMENSIONS; ++k)
      {
        std::cout << _elite_genomes[offset + k] << ", ";
      }
    std::cout << "\n\n" << std::endl;
  }

  template <typename TFloat>
  void ga_solver_gpu<TFloat>::_initialize()
  {
    if(_f_initialized)
      {
        _finalize();
      }

    const uint32_t NUM_ISLES = _dev_population->_NUM_ISLES;
    const uint32_t NUM_AGENTS = _dev_population->_NUM_AGENTS;
    const uint32_t NUM_DIMENSIONS = _dev_population->_NUM_DIMENSIONS;

    uint32_t num_selection = NUM_AGENTS * (NUM_AGENTS - _selection_size);
    num_selection += _selection_p == 0 ?
      0 : NUM_AGENTS * (_selection_size - 1);

    uint32_t num_breed = NUM_AGENTS + NUM_AGENTS * NUM_DIMENSIONS;

    //FIXME: PRNG == agents * eval_prngnumbers?
    uint32_t num_eval_prn = NUM_AGENTS * _evaluator->_num_eval_prnumbers;

    _prn_isle_offset = num_selection + num_breed + num_eval_prn;

    _bulk_size = _prn_isle_offset * NUM_ISLES;

    // Memory resource allocation
    _extended_upper_bounds = new TFloat[NUM_DIMENSIONS];
    _extended_lower_bounds = new TFloat[NUM_DIMENSIONS];

    CudaSafeCall(cudaMalloc((void **) &(_dev_extended_upper_bounds),
                            NUM_DIMENSIONS * sizeof(TFloat)));
    CudaSafeCall(cudaMalloc((void **) &(_dev_extended_lower_bounds),
                            NUM_DIMENSIONS * sizeof(TFloat)));

    CudaSafeCall(cudaMalloc((void **) &(_dev_coupling_idxs),
                            NUM_AGENTS * NUM_ISLES * num_selection * sizeof(uint32_t)));

    CudaSafeCall(cudaMalloc((void **) &(_dev_migrating_idxs),
                            NUM_ISLES * _migration_size * sizeof(uint32_t)));

    CudaSafeCall(cudaMalloc((void **) &(_dev_migration_buffer),
                            NUM_ISLES * (NUM_DIMENSIONS + 1) * _migration_size * sizeof(TFloat)));

    CudaSafeCall(cudaMalloc((void **) &(_dev_bulk_prnumbers),
                            _bulk_size * sizeof(_dev_bulk_prnumbers)));

    _prn_sets[SELECTION_PRNS_OFFSET] = _dev_bulk_prnumbers;
    _prn_sets[BREEDING_PRNS_OFFSET] = _dev_bulk_prnumbers + num_selection;
    _prn_sets[EXTERNAL_PRNS_OFFSET] = _dev_bulk_prnumbers + num_selection + num_breed;

    _elite_genomes = new TFloat [NUM_ISLES * NUM_DIMENSIONS];
    _elite_fitness = new TFloat [NUM_ISLES];

    // Initializing Solver Values
    _generation_count = 0.0;

    for(uint32_t i = 0; i < NUM_ISLES; ++i)
      {
        _elite_fitness[i] = -std::numeric_limits<TFloat>::infinity();
      }

    for(uint32_t i = 0; i < NUM_ISLES; ++i)
      {
        for(uint32_t j = 0; j < NUM_DIMENSIONS; ++j)
          {
            _elite_genomes[i * NUM_DIMENSIONS + j] = std::numeric_limits<TFloat>::quiet_NaN();
          }
      }

    TFloat * const variable_ranges = _dev_population->_var_ranges;
    const TFloat * const upper_bounds = _dev_population->_UPPER_BOUNDS;
    const TFloat * const lower_bounds = _dev_population->_LOWER_BOUNDS;

    // Extended bounds
    for(uint32_t i = 0; i < NUM_DIMENSIONS; ++i)
      {
        // Bound Checking
        const TFloat range = variable_ranges[i];
        const TFloat extension = _range_extension_p == 0 ?
          0 : range * _range_extension_p * 0.5;

        _extended_upper_bounds[i] = upper_bounds[i] + extension;
        _extended_lower_bounds[i] = lower_bounds[i] - extension;

      }

    CudaSafeCall(cudaMemcpy(_dev_extended_upper_bounds,
                            _extended_upper_bounds,
                            sizeof(TFloat) * NUM_DIMENSIONS,
                            cudaMemcpyHostToDevice));

    CudaSafeCall(cudaMemcpy(_dev_extended_lower_bounds,
                            _extended_lower_bounds,
                            sizeof(TFloat) * NUM_DIMENSIONS,
                            cudaMemcpyHostToDevice));

    _f_initialized = true;
  }

  template <typename TFloat>
  void ga_solver_gpu<TFloat>::_finalize()
  {
    if(_f_initialized)
      {
        CudaSafeCall(cudaFree(_dev_extended_upper_bounds));
        CudaSafeCall(cudaFree(_dev_extended_lower_bounds));

        CudaSafeCall(cudaFree(_dev_coupling_idxs));
        CudaSafeCall(cudaFree(_dev_migrating_idxs));
        CudaSafeCall(cudaFree(_dev_migration_buffer));

        CudaSafeCall(cudaFree(_dev_bulk_prnumbers));

        delete [] _extended_upper_bounds;
        delete [] _extended_lower_bounds;

        delete [] _elite_genomes;
        delete [] _elite_fitness;
      }

    _f_initialized = false;
  }

  template <typename TFloat>
  void ga_solver_gpu<TFloat>::_setup_operators(typename ga_operators_gpu<TFloat>::select_func selection_function,
                                               typename ga_operators_gpu<TFloat>::breed_func breeding_function,
                                               typename ga_operators_gpu<TFloat>::migrate_func migration_function)
  {
    _selection_function = selection_function;
    _breeding_function = breeding_function;
    _migration_function = migration_function;
  }

  template <typename TFloat>
  void ga_solver_gpu<TFloat>::_set_migration_config(uint32_t migration_step,
                                                    uint32_t migration_size,
                                                    uint32_t migration_selection_size)
  {
    const uint32_t NUM_AGENTS = _dev_population->_NUM_AGENTS;

    _migration_step = (migration_step > 0) ?
      migration_step : 0;

    _migration_size = (migration_size >= 0 && migration_size <= NUM_AGENTS) ?
      migration_size : 0;

    _migration_selection_size = (migration_selection_size >= 2 &&
                                 migration_selection_size <= NUM_AGENTS) ?
      migration_selection_size : 2;
  }

  template <typename TFloat>
  void ga_solver_gpu<TFloat>::_set_selection_config(uint32_t selection_size,
                                                    TFloat selection_p)
  {
    const uint32_t NUM_AGENTS = _dev_population->_NUM_AGENTS;
    _selection_size = (selection_size >= 2 && selection_size <= NUM_AGENTS) ?
      selection_size : 2;

    _selection_p = (selection_p >= 0.0 && selection_p <= 1.0) ?
      selection_p : 0.0;
  }

  template <typename TFloat>
  void ga_solver_gpu<TFloat>::_set_breeding_config(TFloat crossover_rate,
                                                   TFloat mutation_rate)
  {
    _crossover_rate = (crossover_rate >= 0.0 && crossover_rate <= 1.0) ?
      crossover_rate : 0.9;

    _mutation_rate = (mutation_rate >= 0.0 && mutation_rate <= 1.0) ?
      mutation_rate : 0.1;
  }

  template <typename TFloat>
  void ga_solver_gpu<TFloat>::_set_range_extension(TFloat range_multiplier)
  {
    _range_extension_p = (range_multiplier <= 100 && range_multiplier >= 0) ?
      range_multiplier : 0.0;
  }

  template <typename TFloat>
  void ga_solver_gpu<TFloat>::_set_generation_target(uint32_t generation_target)
  {
    _generation_target = generation_target > 0 ?
      generation_target : 0;
  }


  template <typename TFloat>
  void ga_solver_gpu<TFloat>::_generate_prngs()
  {
    _bulk_prn_generator->_generate(_bulk_size, _dev_bulk_prnumbers);
  }

  template <typename TFloat>
  void ga_solver_gpu<TFloat>::_evaluate_genomes()
  {
    _evaluator->evaluate(_dev_population);
  }

  template <typename TFloat>
  void ga_solver_gpu<TFloat>::_advance_generation()
  {
    if(!_f_initialized)
      {
        std::cerr << "Warning: Uninitialized Solver!\n";
        std::cout << "Initializing Solver...\n";
        _initialize();
      }

    // Renew Pseudo Random Numbers
    _generate_prngs();

    // Evaluate Population's Fitness
    _evaluate_genomes();

    // Update Population Records
    _dev_population->_update_records();

    // Replace Lowest Fitness with elite (Steady-State)
    _replace_update_elite();

    // Population Migration between isles
    if (_generation_count != 0 &&
        _migration_step != 0 &&
        _migration_size != 0 &&
        _generation_count % _migration_step == 0)
      {
        _migrate();
      }

    // Parent Selection
    _select();

    // // Offspring generation: Crossover and Mutation Operation
    _breed();

    // Switch pointers Offspring Genomes <-> Data Genomes
    _dev_population->_swap_data_sets();

    // Advance Counter
    _generation_count++;
  }

  template <typename TFloat>
  void ga_solver_gpu<TFloat>::_select()
  {
    const uint32_t NUM_ISLES = _dev_population->_NUM_ISLES;
    const uint32_t NUM_AGENTS = _dev_population->_NUM_AGENTS;

    const TFloat * const dev_fitness_array = _dev_population->_get_dev_fitness_array();

    _selection_function(false,
                        _selection_size,
                        _selection_p,
                        NUM_ISLES,
                        NUM_AGENTS,
                        dev_fitness_array,
                        _dev_coupling_idxs,
                        _prn_sets[SELECTION_PRNS_OFFSET]);

// #ifdef _DEBUG
//     _coupling_idxs = new uint32_t [NUM_AGENTS];
//     CudaSafeCall(cudaMemcpy(_coupling_idxs,
//                             _dev_coupling_idxs,
//                             NUM_AGENTS * sizeof(uint32_t),
//                             cudaMemcpyDeviceToHost));

//     std::cout << "Selection\n";
//     for(uint32_t i = 0; i < NUM_AGENTS; ++i)
//       {
//         std::cout << _coupling_idxs[i] << std::endl;
//       }
//     std::cout << std::endl;
//     delete [] _coupling_idxs;
// #endif

  }

  template <typename TFloat>
  void ga_solver_gpu<TFloat>::_breed()
  {
    const uint32_t NUM_ISLES = _dev_population->_NUM_ISLES;
    const uint32_t NUM_AGENTS = _dev_population->_NUM_AGENTS;
    const uint32_t NUM_DIMENSIONS = _dev_population->_NUM_DIMENSIONS;

    const TFloat * const ranges = _dev_population->_dev_var_ranges;
    const TFloat * const genomes = _dev_population->_get_dev_data_array();
    TFloat * const offspring = _dev_population->_get_dev_transformed_data_array();

    TFloat generational_ratio = (_generation_target != _generation_target) ?
      1.0 : 1.0 - (_generation_count * 1.0f) / _generation_target * 1.0f;
    generational_ratio = generational_ratio > 0.1 ? generational_ratio : 0.1;

    _breeding_function(_crossover_rate,
                       _mutation_rate,
                       _distribution_iterations,
                       generational_ratio,
                       NUM_ISLES,
                       NUM_AGENTS,
                       NUM_DIMENSIONS,
                       _dev_extended_upper_bounds,
                       _dev_extended_lower_bounds,
                       ranges,
                       genomes,
                       offspring,
                       _dev_coupling_idxs,
                       _prn_sets[BREEDING_PRNS_OFFSET],
                       _bulk_prn_generator);
  }

  template <typename TFloat>
  void ga_solver_gpu<TFloat>::_replace_update_elite()
  {
    const uint32_t NUM_ISLES = _dev_population->_NUM_ISLES;
    const uint32_t NUM_AGENTS = _dev_population->_NUM_AGENTS;
    const uint32_t NUM_DIMENSIONS = _dev_population->_NUM_DIMENSIONS;

    TFloat * const dev_fitness_array = _dev_population->_get_dev_fitness_array();
    TFloat * dev_data_array = _dev_population->_get_dev_data_array();

    const uint32_t * const highest_idx = _dev_population->_get_highest_idx_array();
    const TFloat * const highest_fitness = _dev_population->_get_highest_fitness_array();

    const uint32_t * const lowest_idx = _dev_population->_get_lowest_idx_array();
    const TFloat * const lowest_fitness = _dev_population->_get_lowest_fitness_array();

    // TODO: Replace copy with kernel version
    for(uint32_t i = 0; i < NUM_ISLES; ++i)
      {

// #ifdef _DEBUG
//         std::cout << "ISLE " << i << " LOWEST: " << lowest_fitness[i] << std::endl;
//         std::cout << "ISLE " << i << " HIGHEST: " << highest_fitness[i] << std::endl;
//         std::cout << "ISLE " << i << " ELITE: " << _elite_fitness[i] << std::endl;
// #endif

        // Replace Lowest with Elite iff Lowest < Elite
        {
          if (lowest_fitness[i] < _elite_fitness[i])
            {
              const uint32_t dev_lowest_idx = i * NUM_AGENTS + lowest_idx[i];

              //fitness[local_lowest_idx] = _elite_fitness[i];
              // Copy Elite Fitness -> Lowest Fitness
              CudaSafeCall(cudaMemcpy(dev_fitness_array + dev_lowest_idx,
                                      lowest_fitness + i,
                                      sizeof(TFloat),
                                      cudaMemcpyHostToDevice));

              // Copy Elite Genome -> Lowest Genome
              for(uint32_t j = 0; j < NUM_DIMENSIONS; ++j)
                {
                  CudaSafeCall(cudaMemcpy(dev_data_array + i * NUM_AGENTS * NUM_DIMENSIONS
                                          + lowest_idx[i]
                                          + j * NUM_AGENTS,
                                          _elite_genomes + i * NUM_DIMENSIONS + j,
                                          sizeof(TFloat),
                                          cudaMemcpyHostToDevice));
                }
            }
        }

        // Update Elite iff Highest > Elite
        {
          if (highest_fitness[i] > _elite_fitness[i])
            {
              // Copy Highest Fitness -> Elite Fitness
              const uint32_t dev_highest_idx = i * NUM_ISLES + highest_idx[i];
              CudaSafeCall(cudaMemcpy(_elite_fitness + i,
                                      dev_fitness_array + dev_highest_idx,
                                      sizeof(TFloat),
                                      cudaMemcpyDeviceToHost));

              // Copy Highest Genome -> Elite Genome
              for(uint32_t j = 0; j < NUM_DIMENSIONS; ++j)
                {
                  CudaSafeCall(cudaMemcpy(_elite_genomes + i * NUM_DIMENSIONS + j,
                                          dev_data_array + i * NUM_AGENTS * NUM_DIMENSIONS
                                          + highest_idx[i]
                                          + j * NUM_AGENTS,
                                          sizeof(TFloat),
                                          cudaMemcpyDeviceToHost));
                }
            }
        }
      }
  }

  template <typename TFloat>
  void ga_solver_gpu<TFloat>::_migrate()
  {
    const uint32_t NUM_ISLES = _dev_population->_NUM_ISLES;
    const uint32_t NUM_AGENTS = _dev_population->_NUM_AGENTS;
    const uint32_t NUM_DIMENSIONS = _dev_population->_NUM_DIMENSIONS;

    TFloat * const dev_genomes = _dev_population->_get_dev_data_array();
    TFloat * const dev_fitness_array = _dev_population->_get_dev_fitness_array();

    _migration_function(NUM_ISLES,
                        NUM_AGENTS,
                        NUM_DIMENSIONS,
                        dev_genomes,
                        dev_fitness_array,
                        _dev_migrating_idxs,
                        _dev_migration_buffer,
                        _migration_size,
                        _migration_selection_size,
                        _bulk_prn_generator);

// #ifdef _DEBUG
//     _migrating_idxs = new uint32_t [NUM_ISLES * _migration_size];
//     CudaSafeCall(cudaMemcpy(_migrating_idxs,
//                             _dev_migrating_idxs,
//                             NUM_ISLES * _migration_size * sizeof(uint32_t),
//                             cudaMemcpyDeviceToHost));

//     std::cout << "Migration Selection\n";
//     for(uint32_t i = 0; i < NUM_ISLES; ++i)
//       {
//         for(uint32_t j = 0; j < _migration_size; ++j)
//           {
//             std::cout << _migrating_idxs[i * _migration_size + j] << std::endl;
//           }
//       }
//     std::cout << std::endl;

//     delete [] _migrating_idxs;
// #endif
  }

  template <typename TFloat>
  void ga_solver_gpu<TFloat>::_copy_dev_couples_into_host(uint32_t * const output_couples)
  {
    const uint32_t NUM_ISLES = _population->_NUM_ISLES;
    const uint32_t NUM_AGENTS = _population->_NUM_AGENTS;

    CudaSafeCall(cudaMemcpy(output_couples,
                            _dev_coupling_idxs,
                            sizeof(uint32_t) * NUM_ISLES * NUM_AGENTS,
                            cudaMemcpyDeviceToHost));

    return;
  }

  template <typename TFloat>
  void ga_solver_gpu<TFloat>::_copy_host_couples_into_dev(uint32_t * const input_couples)
  {
    const uint32_t NUM_ISLES = _population->_NUM_ISLES;
    const uint32_t NUM_AGENTS = _population->_NUM_AGENTS;

    CudaSafeCall(cudaMemcpy(_dev_coupling_idxs,
                            input_couples,
                            sizeof(uint32_t) * NUM_ISLES * NUM_AGENTS,
                            cudaMemcpyHostToDevice));

    return;
  }

  template <typename TFloat>
  void ga_solver_gpu<TFloat>::_set_couples_idx(uint32_t * const input_couples)
  {
    _copy_host_couples_into_dev(input_couples);
  }

  template <typename TFloat>
  void ga_solver_gpu<TFloat>::_get_couples_idx(uint32_t * const output_couples)
  {
    _copy_dev_couples_into_host(output_couples);
  }

} // namespace locusta








