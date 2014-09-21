#ifndef LOCUSTA_PSO_SOLVER_GPU_H
#define LOCUSTA_PSO_SOLVER_GPU_H

#include <iostream>

#include "pso_solver.hpp"
#include "pso_operators_gpu.hpp"

#include "../../prngenerator/prngenerator_gpu.hpp"
#include "../../population/population_set_gpu.hpp"

namespace locusta {

  template<typename TFloat>
  class pso_solver_gpu : public pso_solver<TFloat> {

  public:

    pso_solver_gpu(population_set_gpu<TFloat> * population,
                   evaluator_gpu<TFloat> * evaluator,
                   prngenerator_gpu<TFloat> * prn_generator);

    virtual ~pso_solver_gpu();

    virtual void _print_solver_config();

    virtual void _print_solver_elite();

    virtual void _print_solver_solution();

    virtual void _initialize();

    virtual void _finalize();

    virtual void _setup_operators(typename pso_operators<TFloat>::position_func position_function,
                                  typename pso_operators<TFloat>::velocity_func velocity_function,
                                  typename pso_operators<TFloat>::migrate_func migration_function);

    /// Sets up the parent selection operator parameters.
    virtual void _set_pso_config(TFloat inertia_factor,
                                 TFloat cognitive_factor,
                                 TFloat social_factor);

    virtual void _set_velocity_limit_config(TFloat max_velocity,
                                            TFloat min_velocity);

    virtual void _set_range_extension(TFloat range_multiplier);

    virtual void _set_generation_target(uint32_t generation_target);

    virtual void _generate_prngs();

    virtual void _evaluate_genomes();

    virtual void _advance_generation();

    virtual void _update_position();
    virtual void _update_velocity();

    virtual void _migrate();

  protected:

    using pso_solver<TFloat>::_f_initialized;

    using pso_solver<TFloat>::_population;
    /// GPU Population
    population_set_gpu<TFloat> * _dev_population;

    using pso_solver<TFloat>::_evaluator;

    using pso_solver<TFloat>::_bulk_prn_generator;
    using pso_solver<TFloat>::_bulk_prnumbers;
    using pso_solver<TFloat>::_bulk_size;

    using pso_solver<TFloat>::_prn_sets;
    using pso_solver<TFloat>::_prn_isle_offset;

    using pso_solver<TFloat>::_migration_step;
    using pso_solver<TFloat>::_migration_size;
    using pso_solver<TFloat>::_migration_selection_size;
    using pso_solver<TFloat>::_max_velocity;
    using pso_solver<TFloat>::_min_velocity;
    using pso_solver<TFloat>::_inertia_factor;
    using pso_solver<TFloat>::_cognitive_factor;
    using pso_solver<TFloat>::_social_factor;

    using pso_solver<TFloat>::_range_extension_p;

    using pso_solver<TFloat>::_extended_upper_bounds;
    using pso_solver<TFloat>::_extended_lower_bounds;

    using pso_solver<TFloat>::_migrating_idxs;
    using pso_solver<TFloat>::_migration_buffer;
    using pso_solver<TFloat>::_best_fitness;
    using pso_solver<TFloat>::_best_genomes;

    using pso_solver<TFloat>::_update_position;
    using pso_solver<TFloat>::_update_velocity;
    using pso_solver<TFloat>::_migration_function;

    using pso_solver<TFloat>::_generation_count;
    using pso_solver<TFloat>::_generation_target;

    /// GPU Pseudo random numbers array
    TFloat * _dev_bulk_prnumbers;

    /// GPU Migration indexes
    uint32_t * _dev_migrating_idxs;
    /// GPU Migration buffer
    TFloat * _dev_migration_buffer;

  };
} // namespace locusta
#include "pso_solver_gpu_impl.hpp"
#endif
