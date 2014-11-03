#ifndef LOCUSTA_GA_SOLVER_CUDA_H
#define LOCUSTA_GA_SOLVER_CUDA_H

#include "../evolutionary_solver_cuda.hpp"
#include "../../prngenerator/prngenerator_cuda.hpp"

#include "./ga_operators/ga_operators_cuda.hpp"

namespace locusta {

    ///Interface for Genetic Algorithm solver_cudas
    template<typename TFloat>
    struct ga_solver_cuda : evolutionary_solver_cuda<TFloat> {

        enum PRN_OFFSETS { SELECTION_SET = 0, BREEDING_SET = 1 };

        ga_solver_cuda(population_set_cuda<TFloat> * population,
                       evaluator_cuda<TFloat> * evaluator,
                       prngenerator_cuda<TFloat> * prn_generator,
                       uint32_t generation_target,
                       TFloat * upper_bounds,
                       TFloat * lower_bounds);

        virtual ~ga_solver_cuda();

        virtual void setup_solver();

        virtual void teardown_solver();

        virtual void transform();

        virtual void elite_population_replace();

        /// Set Particle Swarm Optimization solver operators.
        virtual void setup_operators(BreedCudaFunctor<TFloat> * breed_functor_ptr,
                                     SelectionCudaFunctor<TFloat> * select_functor_ptr);

        /// Sets up the solver_cuda's configuration
        virtual void solver_config(uint32_t migration_step,
                                   uint32_t migration_size,
                                   uint32_t migration_selection_size,
                                   uint32_t selection_size,
                                   TFloat selection_stochastic_factor,
                                   TFloat crossover_rate,
                                   TFloat mutation_rate,
                                   uint32_t mut_dist_iterations);

        /// Population crossover + mutation operator.
        BreedCudaFunctor<TFloat> * _breed_functor_ptr;

        /// Population couple selection.
        SelectionCudaFunctor<TFloat> * _selection_functor_ptr;

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
        uint32_t * _dev_couples_idx_array;

        /// Temporal candidate selection array
        uint32_t * _dev_candidates_reservoir_array;

        // CUDA specific Evolutionary solver vars
        using evolutionary_solver_cuda<TFloat>::_dev_population;
        using evolutionary_solver_cuda<TFloat>::_dev_evaluator;
        using evolutionary_solver_cuda<TFloat>::_dev_bulk_prn_generator;

        using evolutionary_solver_cuda<TFloat>::_DEV_UPPER_BOUNDS;
        using evolutionary_solver_cuda<TFloat>::_DEV_LOWER_BOUNDS;
        using evolutionary_solver_cuda<TFloat>::_DEV_VAR_RANGES;

        using evolutionary_solver_cuda<TFloat>::_dev_max_agent_genome;
        using evolutionary_solver_cuda<TFloat>::_dev_max_agent_fitness;
        using evolutionary_solver_cuda<TFloat>::_dev_max_agent_idx;

        using evolutionary_solver_cuda<TFloat>::_dev_min_agent_genome;
        using evolutionary_solver_cuda<TFloat>::_dev_min_agent_fitness;
        using evolutionary_solver_cuda<TFloat>::_dev_min_agent_idx;

        using evolutionary_solver_cuda<TFloat>::_dev_migration_idxs;
        using evolutionary_solver_cuda<TFloat>::_dev_migration_buffer;
        using evolutionary_solver_cuda<TFloat>::_dev_bulk_prns;

        // Evolutionary solver vars
        using evolutionary_solver_cuda<TFloat>::_ISLES;
        using evolutionary_solver_cuda<TFloat>::_AGENTS;
        using evolutionary_solver_cuda<TFloat>::_DIMENSIONS;

        using evolutionary_solver_cuda<TFloat>::_UPPER_BOUNDS;
        using evolutionary_solver_cuda<TFloat>::_LOWER_BOUNDS;
        using evolutionary_solver_cuda<TFloat>::_VAR_RANGES;

        using evolutionary_solver_cuda<TFloat>::_population;
        using evolutionary_solver_cuda<TFloat>::_evaluator;

        using evolutionary_solver_cuda<TFloat>::_max_agent_genome;
        using evolutionary_solver_cuda<TFloat>::_max_agent_fitness;
        using evolutionary_solver_cuda<TFloat>::_max_agent_idx;

        using evolutionary_solver_cuda<TFloat>::_min_agent_genome;
        using evolutionary_solver_cuda<TFloat>::_min_agent_fitness;
        using evolutionary_solver_cuda<TFloat>::_min_agent_idx;

        using evolutionary_solver_cuda<TFloat>::_migration_step;
        using evolutionary_solver_cuda<TFloat>::_migration_size;
        using evolutionary_solver_cuda<TFloat>::_migration_selection_size;
        using evolutionary_solver_cuda<TFloat>::_migration_idxs;
        using evolutionary_solver_cuda<TFloat>::_migration_buffer;

        using evolutionary_solver_cuda<TFloat>::_bulk_prn_generator;
        using evolutionary_solver_cuda<TFloat>::_bulk_prns;
        using evolutionary_solver_cuda<TFloat>::_bulk_size;
        using evolutionary_solver_cuda<TFloat>::_prn_sets;

        using evolutionary_solver_cuda<TFloat>::_generation_count;
        using evolutionary_solver_cuda<TFloat>::_generation_target;
        using evolutionary_solver_cuda<TFloat>::_f_initialized;

    };

} // namespace locusta
#include "ga_solver_cuda_impl.hpp"
#endif
