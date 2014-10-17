#ifndef LOCUSTA_PSO_SOLVER_CPU_H
#define LOCUSTA_PSO_SOLVER_CPU_H

#include "../evolutionary_solver_cpu.hpp"
#include "../../prngenerator/prngenerator_cpu.hpp"

#include "./pso_operators/pso_operators.hpp"

namespace locusta {

    ///Interface for Genetic Algorithm solver_cpus
    template<typename TFloat>
    struct pso_solver_cpu : evolutionary_solver_cpu<TFloat> {

        enum class PRN_OFFSETS : uint8_t { COGNITIVE_OFFSET = 0, SOCIAL_OFFSET = 1 };

        pso_solver_cpu(population_set_cpu<TFloat> * population,
                       evaluator_cpu<TFloat> * evaluator,
                       prngenerator_cpu<TFloat> * prn_generator,
                       uint32_t generation_target,
                       TFloat * upper_bounds,
                       TFloat * lower_bounds);

        virtual ~pso_solver_cpu();

        virtual void setup_solver();

        virtual void teardown_solver();

        virtual void transform();

        /// Set Particle Swarm Optimization solver operators.
        virtual void setup_operators(UpdateSpeedFunctor<TFloat> * speed_functor_ptr,
                                     UpdatePositionFunctor<TFloat> * update_functor_ptr);

        /// Sets up the solver_cpu's configuration
        virtual void set_migration_config(uint32_t migration_step,
                                          uint32_t migration_size,
                                          uint32_t migration_selection_size,
                                          TFloat inertia_factor,
                                          TFloat cognitive_factor,
                                          TFloat social_factor);

        /// Particle speed update operator function pointer.
        UpdateSpeedFunctor<TFloat> * _speed_updater_ptr;

        /// Particle position update operator function pointer.
        UpdatePositionFunctor<TFloat> * _position_updater_ptr;

        /// Defines the PSO cognitive factor.
        uint32_t _inertia_factor;

        /// Defines the PSO cognitive factor.
        uint32_t _cognitive_factor;

        /// Defines the PSO social factor.
        TFloat _social_factor;

        /// Describes the best position found per particle.
        TFloat * _cognitive_position_vector;

        /// Describes the velocity vector of each particle.
        TFloat * _velocity_vector;

        /// Describes the locations of each pseudo random number set.
        TFloat * _prn_sets;

        /// Describes the offset within each isle's set.
        std::size_t _prn_isle_offset;

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
        using evolutionary_solver_cpu<TFloat>::_migrating_idxs;
        using evolutionary_solver_cpu<TFloat>::_migration_buffer;

        using evolutionary_solver_cpu<TFloat>::_bulk_prn_generator;
        using evolutionary_solver_cpu<TFloat>::_bulk_prnumbers;
        using evolutionary_solver_cpu<TFloat>::_bulk_size;

        using evolutionary_solver_cpu<TFloat>::_generation_count;
        using evolutionary_solver_cpu<TFloat>::_generation_target;
        using evolutionary_solver_cpu<TFloat>::_f_initialized;

    };

} // namespace locusta
#include "pso_solver_cpu_impl.hpp"
#endif