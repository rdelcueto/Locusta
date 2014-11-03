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
          _max_agent_genome(new TFloat[_ISLES * _DIMENSIONS]),
          _min_agent_genome(new TFloat[_ISLES * _DIMENSIONS]),
          _max_agent_fitness(new TFloat[_ISLES]),
          _min_agent_fitness(new TFloat[_ISLES]),
          _max_agent_idx(new uint32_t[_ISLES]),
          _min_agent_idx(new uint32_t[_ISLES]) {

        for (uint32_t i = 0; i < _DIMENSIONS; i++) {
            _UPPER_BOUNDS[i] = upper_bounds[i];
            _LOWER_BOUNDS[i] = lower_bounds[i];
            _VAR_RANGES[i] = upper_bounds[i] - lower_bounds[i];
        }
    }

    template<typename TFloat>
    evolutionary_solver<TFloat>::~evolutionary_solver() {
        delete [] _UPPER_BOUNDS;
        delete [] _LOWER_BOUNDS;
        delete [] _VAR_RANGES;

        delete [] _max_agent_genome;
        delete [] _min_agent_genome;
        delete [] _max_agent_fitness;
        delete [] _min_agent_fitness;
        delete [] _max_agent_idx;
        delete [] _min_agent_idx;
    }

    template<typename TFloat>
    void evolutionary_solver<TFloat>::setup_solver() {
        // Initialize Population
        if( !_population->_f_initialized ) {
            initialize_vector(_population->_data_array, _population->_transformed_data_array);
            _population->_f_initialized = true;
        }

        // Initialize solver records
        const TFloat * data_array = const_cast<TFloat *>(_population->_data_array);
        for(uint32_t i = 0; i < _ISLES; ++i) {
            for(uint32_t j = 0; j < _AGENTS; ++j) {
                // Initialize best genomes
                _max_agent_idx[i] = 0;
                _min_agent_idx[i] = 0;

                _max_agent_fitness[i] = -std::numeric_limits<TFloat>::infinity();
                _min_agent_fitness[i] = std::numeric_limits<TFloat>::infinity();

                for(uint32_t k = 0; k < _DIMENSIONS; ++k) {
                    const uint32_t data_idx =
                        i * _AGENTS * _DIMENSIONS +
                        0 * _DIMENSIONS +
                        k;

                    _max_agent_genome[i * _DIMENSIONS + k] = data_array[data_idx];
                    _min_agent_genome[i * _DIMENSIONS + k] = data_array[data_idx];
                }
           }
        }

        // Initialize fitness, evaluating initialization data.
        evaluate_genomes();
        // Update solver records.
        update_records();
        // Initialize random numbers.
        regenerate_prns();
    }

    template<typename TFloat>
    void evolutionary_solver<TFloat>::advance() {
        transform();

        _population->swap_data_sets();

        evaluate_genomes();
        update_records();
        regenerate_prns();

        _generation_count++;
    }

    template<typename TFloat>
    void evolutionary_solver<TFloat>::run() {
        do {
            //print_solutions();
            //print_population();
            //print_transformation_diff();
            advance();
        } while(_generation_count % _generation_target != 0);
        print_solutions();
    }

    template<typename TFloat>
    void evolutionary_solver<TFloat>::evaluate_genomes() {
        _evaluator->evaluate(this);
    }

    template<typename TFloat>
    void evolutionary_solver<TFloat>::update_records() {
        TFloat * genomes = _population->_data_array;
        TFloat * fitness = _population->_fitness_array;

        // Scan population
        for(uint32_t i = 0; i < _ISLES; i++) {

            uint32_t isle_max_idx = 0;
            TFloat isle_max_fitness = fitness[i * _AGENTS];
            uint32_t isle_min_idx = 0;
            TFloat isle_min_fitness = fitness[i * _AGENTS];

            for(uint32_t j = 1; j < _AGENTS; j++) {

                const TFloat candidate_fitness = fitness[i * _AGENTS + j];

                if (candidate_fitness > isle_max_fitness) {
                    isle_max_fitness = candidate_fitness;
                    isle_max_idx = j;
                }

                if (candidate_fitness < isle_min_fitness) {
                    isle_min_fitness = candidate_fitness;
                    isle_min_idx = j;
                }
            }

            // Update current isle records
            const TFloat curr_max = _max_agent_fitness[i];

            if (isle_max_fitness > curr_max) {
                _max_agent_idx[i] = isle_max_idx;
                _max_agent_fitness[i] = isle_max_fitness;

                TFloat * max_genome = _max_agent_genome + i * _DIMENSIONS;
                const uint32_t max_genomes_offset = i * _AGENTS * _DIMENSIONS + isle_max_idx * _DIMENSIONS;

                for(uint32_t k = 0; k < _DIMENSIONS; k++) {
                    max_genome[k] = genomes[max_genomes_offset + k];
                }
            }

            const TFloat curr_min = _min_agent_fitness[i];

            if (isle_min_fitness < curr_min) {
                _min_agent_idx[i] = isle_min_idx;
                _min_agent_fitness[i] = isle_min_fitness;

                TFloat * min_genome = _min_agent_genome + i * _DIMENSIONS;
                const uint32_t min_genomes_offset = i * _AGENTS * _DIMENSIONS + isle_min_idx * _DIMENSIONS;

                for(uint32_t k = 0; k < _DIMENSIONS; k++) {
                    min_genome[k] = genomes[min_genomes_offset + k];
                }
            }
       }
    }

    template<typename TFloat>
    void evolutionary_solver<TFloat>::regenerate_prns() {
        _bulk_prn_generator->_generate(_bulk_size, _bulk_prns);
    }

    template<typename TFloat>
    void evolutionary_solver<TFloat>::crop_vector(TFloat * vec) {
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
    void evolutionary_solver<TFloat>::initialize_vector(TFloat * dst_vec, TFloat * tmp_vec) {
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
    void evolutionary_solver<TFloat>::print_transformation_diff() {
        TFloat * fitness = _population->_fitness_array;
        TFloat * genomes = _population->_data_array;
        TFloat * transformed_genomes = _population->_transformed_data_array;

        for(uint32_t i = 0; i < _ISLES; i++) {
            for(uint32_t j = 0; j < _AGENTS; j++) {
                std::cout << fitness[i * _AGENTS + j] << ", [" << i << ", " << j << "] : ";
                TFloat * a = genomes + i * _AGENTS * _DIMENSIONS + j * _DIMENSIONS;
                TFloat * b = transformed_genomes + i * _AGENTS * _DIMENSIONS + j * _DIMENSIONS;
                for(uint32_t k = 0; k < _DIMENSIONS; k++) {
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

    template<typename TFloat>
    void evolutionary_solver<TFloat>::print_population() {
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
    void evolutionary_solver<TFloat>::print_solutions() {
        std::cout << "Solutions @ " << (_generation_count)+1 << " / " << _generation_target << std::endl;
        for(uint32_t i = 0; i < _ISLES; i++) {
            std::cout << _max_agent_fitness[i] << " : [";
            for(uint32_t k = 0; k < _DIMENSIONS; k++) {
                std::cout << _max_agent_genome[i*_DIMENSIONS + k];
                if (k == _DIMENSIONS - 1) {
                    std::cout << "]\n";
                } else {
                    std::cout << ", ";
                }
            }
        }
    }

} // namespace locusta
