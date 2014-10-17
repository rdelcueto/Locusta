#include "evolutionary_solver.hpp"

namespace locusta {

    template<typename TFloat>
    evolutionary_solver<TFloat>::evolutionary_solver(population_set<TFloat> * population,
                                                     evaluator<TFloat> * evaluator,
                                                     prngenerator<TFloat> * prn_generator,
                                                     uint32_t generation_target,
                                                     TFloat * upper_bounds,
                                                     TFloat * lower_bounds)
        : _population(population),
          _evaluator(evaluator),
        _bulk_prn_generator(prn_generator),
        _generation_target(generation_target),
        _generation_count(0),
        _ISLES(population->_ISLES),
        _AGENTS(population->_AGENTS),
        _DIMENSIONS(population->_DIMENSIONS),
        _UPPER_BOUNDS(new TFloat[_DIMENSIONS]),
        _LOWER_BOUNDS(new TFloat[_DIMENSIONS]),
        _VAR_RANGES(new TFloat[_DIMENSIONS]),
        _best_genome(new TFloat[_ISLES * _DIMENSIONS]),
        _best_genome_fitness(new TFloat[_ISLES])
    {
        for (uint32_t i = 0; i < _DIMENSIONS; i++) {
            _UPPER_BOUNDS[i] = upper_bounds[i];
            _LOWER_BOUNDS[i] = lower_bounds[i];
            _VAR_RANGES[i] = upper_bounds[i] - lower_bounds[i];
        }
    }

    template<typename TFloat>
    evolutionary_solver<TFloat>::~evolutionary_solver()
    {
        delete [] _UPPER_BOUNDS;
        delete [] _LOWER_BOUNDS;
        delete [] _VAR_RANGES;

        delete [] _best_genome;
        delete [] _best_genome_fitness;
    }

    template<typename TFloat>
    void evolutionary_solver<TFloat>::setup_solver()
    {
        // Initialize Population
        if( !_population->_f_initialized ) {
            initialize_vector(_population->_data_array, _population->_transformed_data_array);
            _population->_f_initialized = true;
        }

        if( _f_initialized ) {
            teardown_solver();
        }

        // Initialize solver records
        const TFloat * data_array = const_cast<TFloat *>(_population->_data_array);
        for(uint32_t i = 0; i < _ISLES; ++i) {
            for(uint32_t j = 0; j < _AGENTS; ++j) {
                // Initialize best genomes
                for(uint32_t k = 0; k < _DIMENSIONS; ++k) {
                    const uint32_t data_idx =
                        i * _AGENTS * _DIMENSIONS +
                        0 * _DIMENSIONS +
                        k;

                    _best_genome[k] = data_array[data_idx];
                }
                _best_genome_fitness[i] = -std::numeric_limits<TFloat>::infinity();
            }
        }

        // Initialize fitness, evaluating initialization data.
        evaluate_genomes();

        // Update solver records
        update_records();
    }


    template<typename TFloat>
    void evolutionary_solver<TFloat>::advance()
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
    void evolutionary_solver<TFloat>::run()
    {
        do {
            print_solutions();
            advance();
        } while(_generation_count % _generation_target != 0);
    }

    template<typename TFloat>
    void evolutionary_solver<TFloat>::evaluate_genomes()
    {
        _evaluator->evaluate(this);
    }

    template<typename TFloat>
    void evolutionary_solver<TFloat>::update_records()
    {

        TFloat * genomes = _population->_data_array;
        TFloat * fitness = _population->_fitness_array;

        for(uint32_t i = 0; i < _ISLES; i++) {

            uint32_t isle_max_idx = 0;
            TFloat isle_max_fitness = fitness[i * _AGENTS];

            for(uint32_t j = 1; j < _AGENTS; j++) {

                const TFloat candidate_fitness = fitness[i * _AGENTS + j];
                if (candidate_fitness > isle_max_fitness) {
                    isle_max_fitness = candidate_fitness;
                    isle_max_idx = j;
                }
            }

            // Update isle record
            const TFloat current_isle_best = _best_genome_fitness[i];

            if(isle_max_fitness > current_isle_best) {
                _best_genome_fitness[i] = isle_max_fitness;
                TFloat * candidate_genome = _best_genome + i * _DIMENSIONS;
                const uint32_t genomes_offset = i * _AGENTS * _DIMENSIONS + isle_max_idx * _DIMENSIONS;

                for(uint32_t k = 0; k < _DIMENSIONS; k++) {
                    candidate_genome[k] = genomes[genomes_offset + k];
                }
            }
        }
    }

    template<typename TFloat>
    void evolutionary_solver<TFloat>::regenerate_prnumbers()
    {
        _bulk_prn_generator->_generate(_bulk_size, _bulk_prnumbers);
    }

    template<typename TFloat>
    void evolutionary_solver<TFloat>::crop_vector(TFloat * vec)
    {
        // Initialize vector data, within given bounds.
        for(uint32_t i = 0; i < _ISLES; ++i) {
            for(uint32_t j = 0; j < _AGENTS; ++j) {
                for(uint32_t k = 0; k < _DIMENSIONS; ++k) {
                    // Initialize Data Array
                    const uint32_t data_idx =
                        i * _AGENTS * _DIMENSIONS +
                        j * _DIMENSIONS +
                        k;

                    // TODO: Flexible bound crop method
                    TFloat c_value = vec[data_idx];
                    const TFloat value = c_value;
                    const TFloat low_bound = _LOWER_BOUNDS[k];
                    const TFloat high_bound = _UPPER_BOUNDS[k];

                    c_value = c_value < low_bound ? low_bound : c_value;
                    c_value = c_value > high_bound ? high_bound : c_value;

                    if(value != c_value) {
                        // Crop
                        vec[data_idx] = c_value;
                    }
                }
            }
        }
    }

    template<typename TFloat>
    void evolutionary_solver<TFloat>::initialize_vector(TFloat * dst_vec, TFloat * tmp_vec)
    {
        // Initialize vector data, within given bounds.
        const size_t vec_size = _ISLES * _AGENTS * _DIMENSIONS;

        _bulk_prn_generator->_generate(vec_size, tmp_vec);

        for(uint32_t i = 0; i < _ISLES; ++i) {
            for(uint32_t j = 0; j < _AGENTS; ++j) {
                for(uint32_t k = 0; k < _DIMENSIONS; ++k) {
                    // Initialize Data Array
                    const uint32_t data_idx =
                        i * _AGENTS * _DIMENSIONS +
                        j * _DIMENSIONS +
                        k;

                    dst_vec[data_idx] = _LOWER_BOUNDS[k] +
                        (_VAR_RANGES[k] * tmp_vec[data_idx]);
                }
            }
        }
    }

    template<typename TFloat>
    void evolutionary_solver<TFloat>::print_population()
    {
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
    void evolutionary_solver<TFloat>::print_solutions()
    {
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
