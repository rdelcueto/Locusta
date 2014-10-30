#include "ga_solver_cpu.hpp"

namespace locusta {

    ///Interface for Genetic Algorithm solvers
    template<typename TFloat>
    ga_solver_cpu<TFloat>::ga_solver_cpu(population_set_cpu<TFloat> * population,
                                         evaluator_cpu<TFloat> * evaluator,
                                         prngenerator_cpu<TFloat> * prn_generator,
                                         uint32_t generation_target,
                                         TFloat * upper_bounds,
                                         TFloat * lower_bounds)

        : evolutionary_solver_cpu<TFloat>(population,
                                          evaluator,
                                          prn_generator,
                                          generation_target,
                                          upper_bounds,
                                          lower_bounds) {
        // Defaults
        _migration_step = 0;
        _migration_size = 1;
        _migration_selection_size = 2;
        _selection_size = 2;
        _selection_stochastic_factor = 0;
        _crossover_rate = 0.9;
        _mutation_rate = 0.1;
        _mut_dist_iterations = 3;

        // Allocate GA resources
        const size_t TOTAL_GENES = _population->_TOTAL_GENES;
        const size_t TOTAL_AGENTS = _population->_TOTAL_AGENTS;

        _couples_idx_array = new uint32_t[TOTAL_AGENTS];
    }

    template<typename TFloat>
    ga_solver_cpu<TFloat>::~ga_solver_cpu() {
        delete [] _couples_idx_array;
    }

    template<typename TFloat>
    void ga_solver_cpu<TFloat>::setup_solver() {
        // Pseudo random number allocation.
        const uint32_t SELECTION_OFFSET = _selection_functor_ptr->required_prns(this);
        const uint32_t BREEDING_OFFSET = _breed_functor_ptr->required_prns(this);

        _bulk_size = SELECTION_OFFSET + BREEDING_OFFSET;
        _bulk_prns = new TFloat[_bulk_size];

        _prn_sets = new TFloat*[2];
        _prn_sets[SELECTION_SET] = _bulk_prns;
        _prn_sets[BREEDING_SET] = _bulk_prns + SELECTION_OFFSET;

        evolutionary_solver_cpu<TFloat>::setup_solver();
    }

    template<typename TFloat>
    void ga_solver_cpu<TFloat>::teardown_solver() {
        delete [] _prn_sets;
        delete [] _bulk_prns;
    }

    template<typename TFloat>
    void ga_solver_cpu<TFloat>::setup_operators(BreedFunctor<TFloat> * breed_functor_ptr,
                                                SelectionFunctor<TFloat> * selection_functor_ptr) {
        _breed_functor_ptr = breed_functor_ptr;
        _selection_functor_ptr = selection_functor_ptr;
    }

    template<typename TFloat>
    void ga_solver_cpu<TFloat>::solver_config(uint32_t migration_step,
                                              uint32_t migration_size,
                                              uint32_t migration_selection_size,
                                              uint32_t selection_size,
                                              TFloat selection_stochastic_factor,
                                              TFloat crossover_rate,
                                              TFloat mutation_rate,
                                              uint32_t mut_dist_iterations) {
        _migration_step = migration_step;
        _migration_size = migration_size;
        _migration_selection_size = migration_selection_size;
        _selection_size = selection_size;
        _selection_stochastic_factor = selection_stochastic_factor;
        _crossover_rate = crossover_rate;
        _mutation_rate = mutation_rate;
        _mut_dist_iterations = mut_dist_iterations;
    }

    template<typename TFloat>
    void ga_solver_cpu<TFloat>::transform() {
        //elite_population_replace();

        (*_selection_functor_ptr)(this);
        (*_breed_functor_ptr)(this);

        // Crop transformation vector
        evolutionary_solver_cpu<TFloat>::crop_vector(_population->_transformed_data_array);
    }

    template<typename TFloat>
    void ga_solver_cpu<TFloat>::elite_population_replace() {
        TFloat * genomes = _population->_data_array;
        TFloat * fitness = _population->_fitness_array;

        // Scan population
        for(uint32_t i = 0; i < _ISLES; i++) {
            const uint32_t min_idx = _min_agent_idx[i];

            fitness[i * _AGENTS + min_idx] = _max_agent_fitness[i];

            const TFloat * max_genome = _max_agent_genome + i * _DIMENSIONS;
            TFloat * min_genome = genomes + i * _AGENTS * _DIMENSIONS + min_idx * _DIMENSIONS;

            for(uint32_t k = 0; k < _DIMENSIONS; k++) {
                min_genome[k] = max_genome[k];
            }
        }
    }

} // namespace locusta
