#ifndef LOCUSTA_PSO_SOLVER_CPU_H
#define LOCUSTA_PSO_SOLVER_CPU_H

#include "../../prngenerator/prngenerator_cpu.hpp"
#include "../evolutionary_solver_cpu.hpp"

#include "./pso_operators/pso_operators.hpp"

namespace locusta {

/**
 * @brief CPU implementation of the particle swarm optimization solver.
 *
 * This class implements the particle swarm optimization solver for the CPU
 * architecture.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct pso_solver_cpu : evolutionary_solver_cpu<TFloat>
{

  /**
   * @brief Enum defining the offsets for the pseudo-random number sets.
   */
  enum PRN_OFFSETS
  {
    RECORD_UPDATE_SET = 0,
    SPEED_UPDATE_SET = 1,
    POSITION_UPDATE_SET = 2
  };

  /**
   * @brief Construct a new pso_solver_cpu object.
   *
   * @param population Population set.
   * @param evaluator Evaluator.
   * @param prn_generator Pseudo-random number generator.
   * @param generation_target Target number of generations.
   * @param upper_bounds Array of upper bounds for the genes.
   * @param lower_bounds Array of lower bounds for the genes.
   */
  pso_solver_cpu(population_set_cpu<TFloat>* population,
                 evaluator_cpu<TFloat>* evaluator,
                 prngenerator_cpu<TFloat>* prn_generator,
                 uint64_t generation_target,
                 TFloat* upper_bounds,
                 TFloat* lower_bounds);

  /**
   * @brief Destroy the pso_solver_cpu object.
   */
  virtual ~pso_solver_cpu();

  /**
   * @brief Set up the solver.
   *
   * This method initializes and allocates the solver's runtime resources.
   */
  virtual void setup_solver();

  /**
   * @brief Tear down the solver.
   *
   * This method terminates and deallocates the solver's runtime resources.
   */
  virtual void teardown_solver();

  /**
   * @brief Apply the solver's population transformation.
   *
   * This method applies the solver's specific population transformation to
   * generate the next generation of candidate solutions.
   */
  virtual void transform();

  /**
   * @brief Set the particle swarm optimization solver operators.
   *
   * This method sets the particle swarm optimization solver operators,
   * including the particle record updater, speed updater, and position updater.
   *
   * @param update_particle_record_functor_ptr Particle record updater.
   * @param update_speed_functor_ptr Speed updater.
   * @param update_position_functor_ptr Position updater.
   */
  virtual void setup_operators(
    UpdateParticleRecordFunctor<TFloat>* update_particle_record_functor_ptr,
    UpdateSpeedFunctor<TFloat>* update_speed_functor_ptr,
    UpdatePositionFunctor<TFloat>* update_position_functor_ptr);

  /**
   * @brief Configure the solver.
   *
   * This method sets up the solver's configuration, including parameters for
   * migration, inertia, cognitive, and social factors.
   *
   * @param migration_step Migration step size.
   * @param migration_size Migration size.
   * @param migration_selection_size Migration selection size.
   * @param inertia_factor Inertia factor.
   * @param cognitive_factor Cognitive factor.
   * @param social_factor Social factor.
   */
  virtual void solver_config(uint32_t migration_step,
                             uint32_t migration_size,
                             uint32_t migration_selection_size,
                             TFloat inertia_factor,
                             TFloat cognitive_factor,
                             TFloat social_factor);

  /// Particle record update operator function pointer.
  UpdateParticleRecordFunctor<TFloat>* _particle_record_updater_ptr;

  /// Particle speed update operator function pointer.
  UpdateSpeedFunctor<TFloat>* _speed_updater_ptr;

  /// Particle position update operator function pointer.
  UpdatePositionFunctor<TFloat>* _position_updater_ptr;

  /// Defines the PSO cognitive factor.
  TFloat _inertia_factor;

  /// Defines the PSO cognitive factor.
  TFloat _cognitive_factor;

  /// Defines the PSO social factor.
  TFloat _social_factor;

  /// Describes the best position found per particle.
  TFloat* _cognitive_position_vector;

  /// Describes the best position's fitness per particle.
  TFloat* _cognitive_fitness_vector;

  /// Describes the velocity vector of each particle.
  TFloat* _velocity_vector;

  using evolutionary_solver_cpu<TFloat>::_ISLES;
  using evolutionary_solver_cpu<TFloat>::_AGENTS;
  using evolutionary_solver_cpu<TFloat>::_DIMENSIONS;

  using evolutionary_solver_cpu<TFloat>::_UPPER_BOUNDS;
  using evolutionary_solver_cpu<TFloat>::_LOWER_BOUNDS;
  using evolutionary_solver_cpu<TFloat>::_VAR_RANGES;

  using evolutionary_solver_cpu<TFloat>::_population;
  using evolutionary_solver_cpu<TFloat>::_evaluator;

  using evolutionary_solver_cpu<TFloat>::_max_agent_genome;
  using evolutionary_solver_cpu<TFloat>::_max_agent_fitness;
  using evolutionary_solver_cpu<TFloat>::_max_agent_idx;

  using evolutionary_solver_cpu<TFloat>::_min_agent_genome;
  using evolutionary_solver_cpu<TFloat>::_min_agent_fitness;
  using evolutionary_solver_cpu<TFloat>::_min_agent_idx;

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
#include "pso_solver_cpu_impl.hpp"
#endif
