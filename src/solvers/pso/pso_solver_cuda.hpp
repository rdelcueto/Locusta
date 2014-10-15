#ifndef LOCUSTA_PSO_SOLVER_CUDA_H
#define LOCUSTA_PSO_SOLVER_CUDA_H

#include "../evolutionary_solver_cuda.hpp"
#include "../../prngenerator/prngenerator_cuda.hpp"

#include "./pso_operators/pso_operators_cuda.hpp"

namespace locusta {

    ///Interface for Genetic Algorithm solvers
    template<typename TFloat>
    struct pso_solver_cuda : evolutionary_solver_cuda<TFloat> {

        enum class PRN_OFFSETS : uint8_t { COGNITIVE_OFFSET = 0, SOCIAL_OFFSET = 1 };

        pso_solver_cuda(population_set_cuda<TFloat> * population,
                        evaluator_cuda<TFloat> * evaluator,
                        prngenerator_cuda<TFloat> * prn_generator,
                        uint32_t generation_target,
                        TFloat * upper_bounds,
                        TFloat * lower_bounds);

        virtual ~pso_solver_cuda();

        virtual void setup_solver();

        virtual void teardown_solver();

        virtual void transform();

        /// Set Particle Swarm Optimization solver operators.
        virtual void setup_operators(UpdateSpeedCudaFunctor<TFloat> * speed_functor_ptr,
                                     UpdatePositionCudaFunctor<TFloat> * update_functor_ptr);

        /// Sets up the solver's configuration
        virtual void set_migration_config(uint32_t migration_step,
                                          uint32_t migration_size,
                                          uint32_t migration_selection_size,
                                          TFloat inertia_factor,
                                          TFloat cognitive_factor,
                                          TFloat social_factor);

        /// Particle speed update operator function pointer.
        UpdateSpeedCudaFunctor<TFloat> * _speed_updater_ptr;

        /// Particle position update operator function pointer.
        UpdatePositionCudaFunctor<TFloat> * _position_updater_ptr;

        /// Defines the PSO cognitive factor.
        uint32_t _inertia_factor;

        /// Defines the PSO cognitive factor.
        uint32_t _cognitive_factor;

        /// Defines the PSO social factor.
        TFloat _social_factor;

        /// Describes the best position found per particle.
        TFloat * _dev_cognitive_position_vector;

        /// Describes the velocity vector of each particle.
        TFloat * _dev_velocity_vector;

        /// Describes the locations of each pseudo random number set.
        TFloat * _prn_sets;

        /// Describes the offset within each isle's set.
        std::size_t _prn_isle_offset;

        // CUDA specific Evolutionary solver vars
        using evolutionary_solver_cuda<TFloat>::_dev_population;
        using evolutionary_solver_cuda<TFloat>::_dev_evaluator;
        using evolutionary_solver_cuda<TFloat>::_dev_bulk_prn_generator;

        using evolutionary_solver_cuda<TFloat>::_DEV_UPPER_BOUNDS;
        using evolutionary_solver_cuda<TFloat>::_DEV_LOWER_BOUNDS;
        using evolutionary_solver_cuda<TFloat>::_DEV_VAR_RANGES;
        using evolutionary_solver_cuda<TFloat>::_dev_best_genome;
        using evolutionary_solver_cuda<TFloat>::_dev_best_genome_fitness;
        using evolutionary_solver_cuda<TFloat>::_dev_migration_idxs;
        using evolutionary_solver_cuda<TFloat>::_dev_migration_buffer;
        using evolutionary_solver_cuda<TFloat>::_dev_bulk_prnumbers;

        // Evolutionary solver vars
        using evolutionary_solver_cuda<TFloat>::_ISLES;
        using evolutionary_solver_cuda<TFloat>::_AGENTS;
        using evolutionary_solver_cuda<TFloat>::_DIMENSIONS;

        using evolutionary_solver_cuda<TFloat>::_UPPER_BOUNDS;
        using evolutionary_solver_cuda<TFloat>::_LOWER_BOUNDS;
        using evolutionary_solver_cuda<TFloat>::_VAR_RANGES;

        using evolutionary_solver_cuda<TFloat>::_population;
        using evolutionary_solver_cuda<TFloat>::_evaluator;

        using evolutionary_solver_cuda<TFloat>::_best_genome;
        using evolutionary_solver_cuda<TFloat>::_best_genome_fitness;
        using evolutionary_solver_cuda<TFloat>::_migration_step;
        using evolutionary_solver_cuda<TFloat>::_migration_size;
        using evolutionary_solver_cuda<TFloat>::_migration_selection_size;
        using evolutionary_solver_cuda<TFloat>::_migrating_idxs;
        using evolutionary_solver_cuda<TFloat>::_migration_buffer;

        using evolutionary_solver_cuda<TFloat>::_bulk_prn_generator;
        using evolutionary_solver_cuda<TFloat>::_bulk_prnumbers;
        using evolutionary_solver_cuda<TFloat>::_bulk_size;

        using evolutionary_solver_cuda<TFloat>::_generation_count;
        using evolutionary_solver_cuda<TFloat>::_generation_target;
        using evolutionary_solver_cuda<TFloat>::_f_initialized;

    };

} // namespace locusta
#include "pso_solver_cuda_impl.hpp"
#endif
