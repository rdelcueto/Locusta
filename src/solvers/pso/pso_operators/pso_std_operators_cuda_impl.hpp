#ifndef LOCUSTA_PSO_STD_OPERATORS_CUDA_H
#define LOCUSTA_PSO_STD_OPERATORS_CUDA_H

#include "pso_operators_cuda.hpp"

namespace locusta {

/**
 * @brief Dispatch function for the canonical particle record update operator.
 *
 * @param ISLES Number of isles in the population.
 * @param AGENTS Number of agents per isle.
 * @param DIMENSIONS Number of dimensions per agent.
 * @param positions Device array of current particle positions.
 * @param fitness Device array of current particle fitness values.
 * @param record_positions Device array of best known particle positions.
 * @param record_fitness Device array of best known particle fitness values.
 */
template<typename TFloat>
void
canonical_particle_update_dispatch(const uint32_t ISLES,
                                   const uint32_t AGENTS,
                                   const uint32_t DIMENSIONS,
                                   const TFloat* positions,
                                   const TFloat* fitness,
                                   TFloat* record_positions,
                                   TFloat* record_fitness);

/**
 * @brief Dispatch function for the canonical speed update operator.
 *
 * @param ISLES Number of isles in the population.
 * @param AGENTS Number of agents per isle.
 * @param DIMENSIONS Number of dimensions per agent.
 * @param inertia_factor Inertia factor.
 * @param cognitive_factor Cognitive factor.
 * @param social_factor Social factor.
 * @param positions Device array of current particle positions.
 * @param record_positions Device array of best known particle positions.
 * @param isle_record_positions Device array of best known positions for each
 * isle.
 * @param prng_vector Device array of pseudo-random numbers.
 * @param velocities Device array of particle velocities.
 */
template<typename TFloat>
void
canonical_speed_update_dispatch(const uint32_t ISLES,
                                const uint32_t AGENTS,
                                const uint32_t DIMENSIONS,
                                const TFloat inertia_factor,
                                const TFloat cognitive_factor,
                                const TFloat social_factor,
                                const TFloat* positions,
                                const TFloat* record_positions,
                                const TFloat* isle_record_positions,
                                const TFloat* prng_vector,
                                TFloat* velocities);

/**
 * @brief Dispatch function for the canonical position update operator.
 *
 * @param ISLES Number of isles in the population.
 * @param AGENTS Number of agents per isle.
 * @param DIMENSIONS Number of dimensions per agent.
 * @param velocities Device array of particle velocities.
 * @param positions Device array of current particle positions.
 * @param new_positions Device array of new particle positions.
 */
template<typename TFloat>
void
canonical_position_update_dispatch(const uint32_t ISLES,
                                   const uint32_t AGENTS,
                                   const uint32_t DIMENSIONS,
                                   const TFloat* velocities,
                                   const TFloat* positions,
                                   TFloat* new_positions);

/**
 * @brief CUDA implementation of the canonical particle record update operator.
 *
 * This class implements the canonical particle record update operator for the
 * CUDA architecture.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct CanonicalParticleRecordUpdateCuda
  : UpdateParticleRecordCudaFunctor<TFloat>
{

  /**
   * @brief Get the number of pseudo-random numbers required by the operator.
   *
   * @param solver Particle swarm optimization solver.
   * @return Number of pseudo-random numbers required.
   */
  virtual uint32_t required_prns(pso_solver_cuda<TFloat>* solver) { return 0; }

  /**
   * @brief Apply the particle record update operator.
   *
   * @param solver Particle swarm optimization solver.
   */
  void operator()(pso_solver_cuda<TFloat>* solver)
  {
    const uint32_t ISLES = solver->_ISLES;
    const uint32_t AGENTS = solver->_AGENTS;
    const uint32_t DIMENSIONS = solver->_DIMENSIONS;

    const TFloat* positions =
      const_cast<TFloat*>(solver->_dev_population->_dev_data_array);
    const TFloat* fitness =
      const_cast<TFloat*>(solver->_dev_population->_dev_fitness_array);

    TFloat* record_positions = solver->_dev_cognitive_position_vector;
    TFloat* record_fitness = solver->_dev_cognitive_fitness_vector;

    canonical_particle_update_dispatch(ISLES,
                                       AGENTS,
                                       DIMENSIONS,
                                       positions,
                                       fitness,
                                       record_positions,
                                       record_fitness);
  }
};

/**
 * @brief CUDA implementation of the canonical speed update operator.
 *
 * This class implements the canonical speed update operator for the CUDA
 * architecture.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct CanonicalSpeedUpdateCuda : UpdateSpeedCudaFunctor<TFloat>
{

  /**
   * @brief Get the number of pseudo-random numbers required by the operator.
   *
   * @param solver Particle swarm optimization solver.
   * @return Number of pseudo-random numbers required.
   */
  virtual uint32_t required_prns(pso_solver_cuda<TFloat>* solver)
  {
    const uint32_t ISLES = solver->_ISLES;
    const uint32_t AGENTS = solver->_AGENTS;
    const uint32_t DIMENSIONS = solver->_DIMENSIONS;

    return ISLES * AGENTS * DIMENSIONS * 2;
  }

  /**
   * @brief Apply the speed update operator.
   *
   * @param solver Particle swarm optimization solver.
   */
  void operator()(pso_solver_cuda<TFloat>* solver)
  {
    const uint32_t ISLES = solver->_ISLES;
    const uint32_t AGENTS = solver->_AGENTS;
    const uint32_t DIMENSIONS = solver->_DIMENSIONS;

    // Solver configuration constants
    const TFloat inertia_factor = solver->_inertia_factor;
    const TFloat cognitive_factor = solver->_cognitive_factor;
    const TFloat social_factor = solver->_social_factor;

    // Solver state
    const TFloat* positions =
      const_cast<TFloat*>(solver->_dev_population->_dev_data_array); // Current
    // particle position.
    const TFloat* record_positions = const_cast<TFloat*>(
      solver->_dev_cognitive_position_vector); // (Cognitive)
    // Particle's
    // records
    const TFloat* isle_record_positions =
      const_cast<TFloat*>(solver->_dev_max_agent_genome); // (Social)
    // Isle's
    // records
    const TFloat* prng_vector =
      const_cast<TFloat*>(solver->_dev_bulk_prns); // PRNGs

    TFloat* velocities = solver->_dev_velocity_vector; // To be
    // modified

    canonical_speed_update_dispatch(ISLES,
                                    AGENTS,
                                    DIMENSIONS,
                                    inertia_factor,
                                    cognitive_factor,
                                    social_factor,
                                    positions,
                                    record_positions,
                                    isle_record_positions,
                                    prng_vector,
                                    velocities);
  }
};

/**
 * @brief CUDA implementation of the canonical position update operator.
 *
 * This class implements the canonical position update operator for the CUDA
 * architecture.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct CanonicalPositionUpdateCuda : UpdatePositionCudaFunctor<TFloat>
{

  /**
   * @brief Get the number of pseudo-random numbers required by the operator.
   *
   * @param solver Particle swarm optimization solver.
   * @return Number of pseudo-random numbers required.
   */
  virtual uint32_t required_prns(pso_solver_cuda<TFloat>* solver) { return 0; }

  /**
   * @brief Apply the position update operator.
   *
   * @param solver Particle swarm optimization solver.
   */
  void operator()(pso_solver_cuda<TFloat>* solver)
  {
    const uint32_t ISLES = solver->_ISLES;
    const uint32_t AGENTS = solver->_AGENTS;
    const uint32_t DIMENSIONS = solver->_DIMENSIONS;

    const TFloat* velocities =
      const_cast<TFloat*>(solver->_dev_velocity_vector);
    const TFloat* positions =
      const_cast<TFloat*>(solver->_dev_population->_dev_data_array);

    TFloat* new_positions =
      solver->_dev_population->_dev_transformed_data_array;

    canonical_position_update_dispatch(
      ISLES, AGENTS, DIMENSIONS, velocities, positions, new_positions);
  }
};
}

#endif
