namespace locusta {

    template <typename TFloat>
    ga_solver_cpu<TFloat>::ga_solver_cpu (population_set_cpu<TFloat> * population,
                                          evaluator_cpu<TFloat> * evaluator,
                                          prngenerator_cpu<TFloat> * prn_generator)
        : ga_solver<TFloat>(population,
                            evaluator,
                            prn_generator)
    {
    }

    template <typename TFloat>
    ga_solver_cpu<TFloat>::~ga_solver_cpu()
    {
    }

    template <typename TFloat>
    void ga_solver_cpu<TFloat>::_print_solver_config()
    {
        std::cout << "\nGenetic Algorithm Solver (CPU)"
                  << "\nConfiguration:"
                  << "\n\t * Isles / Agents / Dimensions: "
                  <<  _population->_NUM_ISLES
                  << " / " <<  _population->_NUM_AGENTS
                  << " / " << _population->_NUM_DIMENSIONS
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
    void ga_solver_cpu<TFloat>::_print_solver_elite()
    {
        const uint32_t NUM_DIMENSIONS = _population->_NUM_DIMENSIONS;
        const uint32_t NUM_ISLES = _population->_NUM_ISLES;

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
    void ga_solver_cpu<TFloat>::_print_solver_solution()
    {
        const uint32_t NUM_ISLES = _population->_NUM_ISLES;
        const uint32_t NUM_DIMENSIONS = _population->_NUM_DIMENSIONS;

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
    void ga_solver_cpu<TFloat>::_initialize()
    {
        if(_f_initialized)
        {
            _finalize();
        }

        const uint32_t NUM_ISLES = _population->_NUM_ISLES;
        const uint32_t NUM_AGENTS = _population->_NUM_AGENTS;
        const uint32_t NUM_DIMENSIONS = _population->_NUM_DIMENSIONS;

        uint32_t num_selection = NUM_AGENTS * (NUM_AGENTS - _selection_size);
        num_selection +=_selection_p == 0 ?
            0 : NUM_AGENTS * _selection_size;

        uint32_t num_breed = NUM_AGENTS + NUM_AGENTS * NUM_DIMENSIONS;

        ///FIXME: PRNG == agents * eval_prngnumbers?
        uint32_t num_eval_prn = NUM_AGENTS * _evaluator->_num_eval_prnumbers;

        _prn_isle_offset = num_selection + num_breed + num_eval_prn;

        _bulk_size = _prn_isle_offset * NUM_ISLES;

        /// Memory resource allocation
        _extended_upper_bounds = new TFloat[NUM_DIMENSIONS];
        _extended_lower_bounds = new TFloat[NUM_DIMENSIONS];
        _coupling_idxs = new uint32_t [NUM_AGENTS * NUM_ISLES];

        _migrating_idxs = new uint32_t [NUM_ISLES * _migration_size];
        _migration_buffer = new TFloat [NUM_ISLES * (NUM_DIMENSIONS + 1) * _migration_size];

        _bulk_prnumbers = new  TFloat [_bulk_size];

        _prn_sets[SELECTION_PRNS_OFFSET] = _bulk_prnumbers;
        _prn_sets[BREEDING_PRNS_OFFSET] = _bulk_prnumbers + num_selection;
        _prn_sets[EXTERNAL_PRNS_OFFSET] = _bulk_prnumbers + num_selection + num_breed;

        _elite_genomes = new TFloat [NUM_ISLES * NUM_DIMENSIONS];
        _elite_fitness = new TFloat [NUM_ISLES];

        /// Initializing Solver Values
        _generation_count = 0.0;

        for(uint32_t i = 0; i < NUM_ISLES; ++i)
        {
            _elite_fitness[i] = -std::numeric_limits<TFloat>::infinity();
        }

        TFloat * const variable_ranges = _population->_var_ranges;
        const TFloat * const upper_bounds = _population->_UPPER_BOUNDS;
        const TFloat * const lower_bounds = _population->_LOWER_BOUNDS;

        /// Extended bounds
        for(uint32_t i = 0; i < NUM_DIMENSIONS; ++i)
        {
            /// Bound Checking
            const TFloat range = variable_ranges[i];
            const TFloat extension = _range_extension_p == 0 ?
                0 : range * _range_extension_p * 0.5;

            _extended_upper_bounds[i] = upper_bounds[i] + extension;
            _extended_lower_bounds[i] = lower_bounds[i] - extension;
        }

#ifdef _DEBUG
        std::cout << "\nOriginal Upper Bounds\n";
        for(uint32_t i = 0; i < NUM_DIMENSIONS; ++i)
        {
            std::cout << upper_bounds[i] << ", ";
        }

        std::cout << "\nOriginal Lower Bounds\n";
        for(uint32_t i = 0; i < NUM_DIMENSIONS; ++i)
        {
            std::cout << lower_bounds[i] << ", ";
        }

        std::cout << "\nExtended Upper Bounds\n";
        for(uint32_t i = 0; i < NUM_DIMENSIONS; ++i)
        {
            std::cout << _extended_upper_bounds[i] << ", ";
        }

        std::cout << "\nExtended Lower Bounds\n";
        for(uint32_t i = 0; i < NUM_DIMENSIONS; ++i)
        {
            std::cout << _extended_lower_bounds[i] << ", ";
        }
        std::cout << std::endl;
#endif

        _f_initialized = true;
    }

    template <typename TFloat>
    void ga_solver_cpu<TFloat>::_finalize()
    {
        if(_f_initialized)
        {
            delete [] _extended_upper_bounds;
            delete [] _extended_lower_bounds;
            delete [] _coupling_idxs;

            delete [] _migrating_idxs;
            delete [] _migration_buffer;
            delete [] _bulk_prnumbers;
            delete [] _elite_genomes;
            delete [] _elite_fitness;
        }

        _f_initialized = false;
    }

    template <typename TFloat>
    void ga_solver_cpu<TFloat>::_initialize_population()
    {
        TFloat * const pop_data = _population->_get_transformed_data_array();
        const uint32_t pop_size = _population->_NUM_DIMENSIONS * _population->_NUM_AGENTS * _population->_NUM_ISLES;

        _bulk_prn_generator->_generate(pop_size, pop_data);
        _population->_initialize();
    }

    template <typename TFloat>
    void ga_solver_cpu<TFloat>::_setup_operators(typename ga_operators_cpu<TFloat>::select_func selection_function,
                                                 typename ga_operators_cpu<TFloat>::breed_func breeding_function,
                                                 typename ga_operators_cpu<TFloat>::migrate_func migration_function)
    {
        _selection_function = selection_function;
        _breeding_function = breeding_function;
        _migration_function = migration_function;
    }

    template <typename TFloat>
    void ga_solver_cpu<TFloat>::_set_migration_config(uint32_t migration_step,
                                                      uint32_t migration_size,
                                                      uint32_t migration_selection_size)
    {
        const uint32_t NUM_AGENTS = _population->_NUM_AGENTS;

        _migration_step = (migration_step > 0) ?
            migration_step : 0;

        _migration_size = (migration_size >= 0 && migration_size <= NUM_AGENTS) ?
            migration_size : 0;

        _migration_selection_size = (migration_selection_size >= 2 &&
                                     migration_selection_size <= NUM_AGENTS) ?
            migration_selection_size : 2;
    }

    template <typename TFloat>
    void ga_solver_cpu<TFloat>::_set_selection_config(uint32_t selection_size,
                                                      TFloat selection_p)
    {
        const uint32_t NUM_AGENTS = _population->_NUM_AGENTS;
        _selection_size = (selection_size >= 2 && selection_size <= NUM_AGENTS) ?
            selection_size : 2;

        _selection_p = (selection_p >= 0.0 && selection_p <= 1.0) ?
            selection_p : 0.0;
    }

    template <typename TFloat>
    void ga_solver_cpu<TFloat>::_set_breeding_config(TFloat crossover_rate,
                                                     TFloat mutation_rate)
    {
        _crossover_rate = (crossover_rate >= 0.0 && crossover_rate <= 1.0) ?
            crossover_rate : 0.9;

        _mutation_rate = (mutation_rate >= 0.0 && mutation_rate <= 1.0) ?
            mutation_rate : 0.1;
    }

    template <typename TFloat>
    void ga_solver_cpu<TFloat>::_set_range_extension(TFloat range_multiplier)
    {
        _range_extension_p = (range_multiplier <= 100 && range_multiplier >= 0) ?
            range_multiplier : 0.0;
    }

    template <typename TFloat>
    void ga_solver_cpu<TFloat>::_set_generation_target(uint32_t generation_target)
    {
        _generation_target = generation_target > 0 ?
            generation_target : 0;
    }

    template <typename TFloat>
    void ga_solver_cpu<TFloat>::_generate_prngs()
    {
        _bulk_prn_generator->_generate(_bulk_size, _bulk_prnumbers);
    }

    template <typename TFloat>
    void ga_solver_cpu<TFloat>::_evaluate_genomes()
    {
        _evaluator->evaluate(_population);
    }

    template <typename TFloat>
    void ga_solver_cpu<TFloat>::_advance_generation()
    {
        if(!_f_initialized)
        {
            std::cerr << "Warning: Uninitialized Solver!\n";
            std::cout << "Initializing Solver...\n";
            _initialize();
        }

        /// Renew Pseudo Random Numbers
        _generate_prngs();

        /// Evaluate Population's Fitness
        _evaluate_genomes();

        /// Update Population Records
        _population->_update_records();

        /// Replace Lowest Fitness with elite (Steady-State)
        _replace_update_elite();

        /// Population Migration between isles
        if (this->_generation_count != 0 &&
            this->_migration_step != 0 &&
            this->_migration_size != 0 &&
            this->_generation_count % this->_migration_step == 0)
        {
            _migrate();
        }

        /// Parent Selection
        _select();

        /// Offspring generation: Crossover and Mutation Operation
        _breed();

        /// Switch pointers Offspring Genomes <-> Data Genomes
        _population->_swap_data_sets();

        /// Advance Counter
        _generation_count++;
    }

    template <typename TFloat>
    void ga_solver_cpu<TFloat>::_select()
    {
        const uint32_t NUM_ISLES = _population->_NUM_ISLES;
        const uint32_t NUM_AGENTS = _population->_NUM_AGENTS;

        const TFloat * const fitness_array = _population->_get_fitness_array();

        _selection_function(false,
                            _selection_size,
                            _selection_p,
                            NUM_ISLES,
                            NUM_AGENTS,
                            fitness_array,
                            _coupling_idxs,
                            _prn_sets[SELECTION_PRNS_OFFSET]);
    }

    template <typename TFloat>
    void ga_solver_cpu<TFloat>::_breed()
    {
        const uint32_t NUM_ISLES = _population->_NUM_ISLES;
        const uint32_t NUM_AGENTS = _population->_NUM_AGENTS;
        const uint32_t NUM_DIMENSIONS = _population->_NUM_DIMENSIONS;

        const TFloat * const ranges = _population->_var_ranges;
        const TFloat * const genomes = _population->_get_data_array();
        TFloat * const offspring = _population->_get_transformed_data_array();

        TFloat generational_ratio = (this->_generation_target != this->_generation_target) ?
            1.0 : 1.0 - (this->_generation_count * 1.0f) / this->_generation_target * 1.0f;
        generational_ratio = generational_ratio > 0.1 ? generational_ratio : 0.1;

#ifdef _DEBUG
        std::cout << "GR: " << generational_ratio << std::endl;
#endif

#pragma omp parallel for schedule(static)
        for(uint32_t i = 0; i < NUM_ISLES; ++i)
        {
            ///const int nthread = omp_get_thread_num();
            const uint32_t isle_var_offset = i * NUM_AGENTS * NUM_DIMENSIONS;
            const uint32_t isle_couple_offset = i * NUM_AGENTS;

            _breeding_function(_crossover_rate,
                               _mutation_rate,
                               _distribution_iterations,
                               generational_ratio,
                               NUM_ISLES,
                               NUM_AGENTS,
                               NUM_DIMENSIONS,
                               _extended_upper_bounds,
                               _extended_lower_bounds,
                               ranges,
                               genomes + isle_var_offset,
                               offspring + isle_var_offset,
                               _coupling_idxs + isle_couple_offset,
                               _prn_sets[BREEDING_PRNS_OFFSET] + i * _prn_isle_offset,
                               _bulk_prn_generator);
        }
    }

    template <typename TFloat>
    void ga_solver_cpu<TFloat>::_replace_update_elite()
    {
        const uint32_t NUM_ISLES = _population->_NUM_ISLES;
        ///const uint32_t NUM_AGENTS = _population->_NUM_AGENTS;
        const uint32_t NUM_DIMENSIONS = _population->_NUM_DIMENSIONS;

        TFloat * const fitness = _population->_get_fitness_array();
        TFloat * data_array = _population->_get_data_array();

        const uint32_t * const highest_idx = _population->_get_highest_idx_array();
        const TFloat * const highest_fitness = _population->_get_highest_fitness_array();

        const uint32_t * const lowest_idx = _population->_get_lowest_idx_array();
        TFloat * const lowest_fitness = _population->_get_lowest_fitness_array();

        /// #pragma omp parallel for schedule(static)
        for(uint32_t i = 0; i < NUM_ISLES; ++i)
        {
            /// Replace Lowest with Elite iff Lowest < Elite
            {
                if (lowest_fitness[i] < _elite_fitness[i])
                {
                    /// Copy Elite Fitness -> Lowest Fitness
                    lowest_fitness[i] = _elite_fitness[i]; /// TODO: CHECK idx

                    TFloat * const local_lowest_genome = data_array + lowest_idx[i] * NUM_DIMENSIONS;
                    TFloat * const local_elite_genome = _elite_genomes + i * NUM_DIMENSIONS;

                    /// Copy Elite Genome -> Lowest Genome
                    for(uint32_t k = 0; k < NUM_DIMENSIONS; ++k)
                    {
                        local_lowest_genome[k] = local_elite_genome[k];
                    }
                }
            }

            /// Update Elite iff Highest > Elite
            {
                if (highest_fitness[i] > _elite_fitness[i])
                {
                    /// Copy Highest Fitness -> Elite Fitness
                    _elite_fitness[i] = fitness[highest_idx[i]];

                    TFloat * const local_highest_genome = data_array + highest_idx[i] * NUM_DIMENSIONS;
                    TFloat * const local_elite_genome = _elite_genomes + i * NUM_DIMENSIONS;

                    /// Copy Highest Genome -> Elite Genome
                    for(uint32_t k = 0; k < NUM_DIMENSIONS; ++k)
                    {
                        local_elite_genome[k] = local_highest_genome[k];
                    }
                }
            }
        }
    }

    template <typename TFloat>
    void ga_solver_cpu<TFloat>::_migrate()
    {
        const uint32_t NUM_ISLES = _population->_NUM_ISLES;
        const uint32_t NUM_AGENTS = _population->_NUM_AGENTS;
        const uint32_t NUM_DIMENSIONS = _population->_NUM_DIMENSIONS;

        TFloat * const genomes = _population->_get_data_array();
        TFloat * const fitness_array = _population->_get_fitness_array();

        /// Migration Genome Exchange
        _migration_function(NUM_ISLES,
                            NUM_AGENTS,
                            NUM_DIMENSIONS,
                            genomes,
                            fitness_array,
                            _migrating_idxs,
                            _migration_buffer,
                            _migration_size,
                            _migration_selection_size,
                            _bulk_prn_generator);
    }

} /// namespace locusta
