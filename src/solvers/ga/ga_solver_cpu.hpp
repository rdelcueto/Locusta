#ifndef LOCUSTA_GA_SOLVER_CPU_H
#define LOCUSTA_GA_SOLVER_CPU_H

#include "../evolutionary_solver_cpu.hpp"
#include "../../prngenerator/prngenerator_cpu.hpp"

#include "./ga_operators/ga_operators.hpp"

namespace locusta {

    ///Interface for Genetic Algorithm solver_cpus
    template<typename TFloat>
    struct ga_solver_cpu : evolutionary_solver_cpu<TFloat> {

        enum PRN_OFFSETS { SELECTION_SET = 0, BREEDING_SET = 1 };

        ga_solver_cpu(population_set_cpu<TFloat> * population,
                      evaluator_cpu<TFloat> * evaluator,
                      prngenerator_cpu<TFloat> * prn_generator,
                      uint32_t generation_target,
                      TFloat * upper_bounds,
                      TFloat * lower_bounds);

        virtual ~ga_solver_cpu();

        virtual void setup_solver();

        virtual void teardown_solver();

        virtual void transform();

        /// Set Particle Swarm Optimization solver operators.
        virtual void setup_operators(BreedFunctor<TFloat> * breed_functor_ptr,
                                     SelectionFunctor<TFloat> * select_functor_ptr);

        /// Sets up the solver_cpu's configuration
        virtual void solver_config(uint32_t migration_step,
                                   uint32_t migration_size,
                                   uint32_t migration_selection_size,
                                   uint32_t selection_size,
                                   TFloat selection_stochastic_factor,
                                   TFloat crossover_rate,
                                   TFloat mutation_rate,
                                   uint32_t mut_dist_iterations);

        /// Population crossover + mutation operator.
        BreedFunctor<TFloat> * _breed_functor_ptr;

        /// Population couple selection.
        SelectionFunctor<TFloat> * _selection_functor_ptr;

        /// Tournament selection size.
        uint32_t _selection_size;

        /// Tournament stochastic factor
        TFloat _selection_stochastic_factor;

        /// Crossover rate.
        TFloat _crossover_rate;

        /// Mutation rate.
        TFloat _mutation_rate;

        /// Mutation operator, distribution operator.
        uint32_t _mut_dist_iterations;

        /// Couple selection array.
        uint32_t * _couples_idx_array;

        using evolutionary_solver_cpu<TFloat>::_ISLES;
        using evolutionary_solver_cpu<TFloat>::_AGENTS;
        using evolutionary_solver_cpu<TFloat>::_DIMENSIONS;

        using evolutionary_solver_cpu<TFloat>::_UPPER_BOUNDS;
        using evolutionary_solver_cpu<TFloat>::_LOWER_BOUNDS;
        using evolutionary_solver_cpu<TFloat>::_VAR_RANGES;

        using evolutionary_solver_cpu<TFloat>::_population;
        using evolutionary_solver_cpu<TFloat>::_evaluator;

        using evolutionary_solver_cpu<TFloat>::_best_genome;
        using evolutionary_solver_cpu<TFloat>::_best_genome_fitness;

        using evolutionary_solver_cpu<TFloat>::_migration_step;
        using evolutionary_solver_cpu<TFloat>::_migration_size;
        using evolutionary_solver_cpu<TFloat>::_migration_selection_size;
        using evolutionary_solver_cpu<TFloat>::_migration_idxs;
        using evolutionary_solver_cpu<TFloat>::_migration_buffer;

        using evolutionary_solver_cpu<TFloat>::_bulk_prn_generator;
        using evolutionary_solver_cpu<TFloat>::_bulk_prns;
        using evolutionary_solver_cpu<TFloat>::_bulk_size;
        using evolutionary_solver_cpu<TFloat>::_prn_sets;

        using evolutionary_solver_cpu<TFloat>::_generation_count;
        using evolutionary_solver_cpu<TFloat>::_generation_target;
        using evolutionary_solver_cpu<TFloat>::_f_initialized;

    };

} // namespace locusta
#include "ga_solver_cpu_impl.hpp"
#endif
