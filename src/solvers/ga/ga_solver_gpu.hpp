#ifndef LOCUSTA_GA_SOLVER_GPU_H
#define LOCUSTA_GA_SOLVER_GPU_H

#include <iostream>

#include "ga_solver.hpp"
#include "ga_operators_gpu.hpp"

#include "../../prngenerator/prngenerator_gpu.hpp"
#include "../../population/population_set_gpu.hpp"

namespace locusta {

  template<typename TFloat>
  class ga_solver_gpu : public ga_solver<TFloat> {

  public:

    ga_solver_gpu(population_set_gpu<TFloat> * population,
                  evaluator_gpu<TFloat> * evaluator,
                  prngenerator_gpu<TFloat> * prn_generator);

    virtual ~ga_solver_gpu();

    virtual void _print_solver_config();

    virtual void _print_solver_elite();

    virtual void _print_solver_solution();

    virtual void _initialize();

    virtual void _finalize();

    virtual void _setup_operators(typename ga_operators_gpu<TFloat>::select_func selection_function,
                                  typename ga_operators_gpu<TFloat>::breed_func breeding_function,
                                  typename ga_operators_gpu<TFloat>::migrate_func migration_function);

    virtual void _set_migration_config(uint32_t migration_step,
                                       uint32_t migration_size,
                                       uint32_t migration_selection_size);

    virtual void _set_selection_config(uint32_t selection_size,
                                       TFloat selection_p);

    virtual void _set_breeding_config(TFloat crossover_rate,
                                      TFloat mutation_rate);

    virtual void _set_range_extension(TFloat range_multiplier);

    virtual void _set_generation_target(uint32_t generation_target);

    virtual void _generate_prngs();

    virtual void _evaluate_genomes();

    virtual void _advance_generation();

  protected:

    virtual void _select();
    virtual void _breed();
    virtual void _replace_update_elite();
    virtual void _migrate();

    using ga_solver<TFloat>::_f_initialized;

    using ga_solver<TFloat>::_population;
    population_set_gpu<TFloat> * _dev_population;

    using ga_solver<TFloat>::_evaluator;

    using ga_solver<TFloat>::_bulk_prn_generator;
    using ga_solver<TFloat>::_bulk_prnumbers;
    using ga_solver<TFloat>::_bulk_size;

    using ga_solver<TFloat>::_prn_sets;
    using ga_solver<TFloat>::_prn_isle_offset;

    using ga_solver<TFloat>::_migration_step;
    using ga_solver<TFloat>::_migration_size;
    using ga_solver<TFloat>::_migration_selection_size;
    using ga_solver<TFloat>::_selection_size;
    using ga_solver<TFloat>::_selection_p;
    using ga_solver<TFloat>::_crossover_rate;
    using ga_solver<TFloat>::_mutation_rate;
    using ga_solver<TFloat>::_distribution_iterations;
    using ga_solver<TFloat>::_range_extension_p;

    using ga_solver<TFloat>::_extended_upper_bounds;
    using ga_solver<TFloat>::_extended_lower_bounds;
    using ga_solver<TFloat>::_coupling_idxs;
    using ga_solver<TFloat>::_migrating_idxs;

    using ga_solver<TFloat>::_elite_fitness;
    using ga_solver<TFloat>::_elite_genomes;

    using ga_solver<TFloat>::_selection_function;
    using ga_solver<TFloat>::_breeding_function;
    using ga_solver<TFloat>::_migration_function;

    using ga_solver<TFloat>::_generation_count;
    using ga_solver<TFloat>::_generation_target;

    TFloat * _dev_bulk_prnumbers;

    TFloat * _dev_extended_upper_bounds;
    TFloat * _dev_extended_lower_bounds;
    uint32_t * _dev_coupling_idxs;
    uint32_t * _dev_migrating_idxs;
    TFloat * _dev_migration_buffer;

  };
} /// namespace locusta
#include "ga_solver_gpu_impl.hpp"
#endif
