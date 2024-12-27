#ifndef LOCUSTA_PSO_STD_OPERATORS_H
#define LOCUSTA_PSO_STD_OPERATORS_H

#include "prngenerator/prngenerator_cpu.hpp"
#include "pso_operators.hpp"

namespace locusta {

/**
 * @brief Canonical particle record update operator.
 *
 * This class implements the canonical particle record update operator, which
 * updates the best known position and fitness for each particle.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct CanonicalParticleRecordUpdate : UpdateParticleRecordFunctor<TFloat>
{

  /**
   * @brief Get the number of pseudo-random numbers required by the operator.
   *
   * @param solver Particle swarm optimization solver.
   * @return Number of pseudo-random numbers required.
   */
  uint32_t required_prns(pso_solver_cpu<TFloat>* solver) { return 0; }

  /**
   * @brief Apply the particle record update operator.
   *
   * @param solver Particle swarm optimization solver.
   */
  void operator()(pso_solver_cpu<TFloat>* solver)
  {
    const uint32_t ISLES = solver->_ISLES;
    const uint32_t AGENTS = solver->_AGENTS;
    const uint32_t DIMENSIONS = solver->_DIMENSIONS;

    const TFloat* position_array =
      const_cast<TFloat*>(solver->_population->_data_array);

    const TFloat* fitness_array =
      const_cast<TFloat*>(solver->_population->_fitness_array);

    TFloat* position_record = solver->_cognitive_position_vector; // (Cognitive)
    TFloat* fitness_record = solver->_cognitive_fitness_vector;   // (Cognitive
                                                                  // Fitness)
#pragma omp parallel for collapse(2)
    for (uint32_t i = 0; i < ISLES; i++) {
      for (uint32_t j = 0; j < AGENTS; j++) {
        const uint32_t agent_idx = i * AGENTS + j;
        const TFloat curr_fitness = fitness_array[agent_idx];
        const TFloat candidate_fitness = fitness_record[agent_idx];
        if (curr_fitness > candidate_fitness) {
          fitness_record[agent_idx] = curr_fitness;
          const uint32_t genome_offset =
            i * AGENTS * DIMENSIONS + j * DIMENSIONS;
          for (uint32_t k = 0; k < DIMENSIONS; k++) {
            const uint32_t curr_gene = genome_offset + k;
            position_record[curr_gene] = position_array[curr_gene];
          }
        }
      }
    }
  }
};

/**
 * @brief Canonical speed update operator.
 *
 * This class implements the canonical speed update operator, which updates the
 * speed of each particle based on its current position, best known position,
 * and the best known position of its neighbors.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct CanonicalSpeedUpdate : UpdateSpeedFunctor<TFloat>
{
  /**
   * @brief Get the number of pseudo-random numbers required by the operator.
   *
   * @param solver Particle swarm optimization solver.
   * @return Number of pseudo-random numbers required.
   */
  uint32_t required_prns(pso_solver_cpu<TFloat>* solver)
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
  void operator()(pso_solver_cpu<TFloat>* solver)
  {
    const uint32_t ISLES = solver->_ISLES;
    const uint32_t AGENTS = solver->_AGENTS;
    const uint32_t DIMENSIONS = solver->_DIMENSIONS;

    // Solver configuration constants
    const TFloat inertia_factor = solver->_inertia_factor;
    const TFloat cognitive_factor = solver->_cognitive_factor;
    const TFloat social_factor = solver->_social_factor;

    // Solver state
    TFloat* velocities = solver->_velocity_vector; // To be modified
    const TFloat* positions =
      const_cast<TFloat*>(solver->_population->_data_array); // Current
    // particle position.
    const TFloat* position_record =
      const_cast<TFloat*>(solver->_cognitive_position_vector); // (Cognitive)
    // Particle's
    // records
    const TFloat* isle_position_record =
      const_cast<TFloat*>(solver->_max_agent_genome); // (Social)
    // Isle's
    // records

    const uint32_t RND_OFFSET = DIMENSIONS * 2;
    const TFloat* prn_array = const_cast<TFloat*>(
      solver->_prn_sets[pso_solver_cpu<TFloat>::SPEED_UPDATE_SET]);

#pragma omp parallel for collapse(2)
    for (uint32_t i = 0; i < ISLES; i++) {
      for (uint32_t j = 0; j < AGENTS; j++) {
        const uint32_t isle_position_record_offset = i * DIMENSIONS;
        const uint32_t particle_offset =
          i * AGENTS * DIMENSIONS + j * DIMENSIONS;
        const TFloat* agents_prns =
          prn_array + i * AGENTS * RND_OFFSET + j * RND_OFFSET;

#pragma omp simd
        for (uint32_t k = 0; k < DIMENSIONS; k++) {
          const uint32_t prn_idx = k * 2;
          const TFloat c_rnd = agents_prns[prn_idx];
          const TFloat s_rnd = agents_prns[prn_idx + 1];

          const TFloat p_g =
            isle_position_record[isle_position_record_offset + k];
          const TFloat p_i = position_record[particle_offset + k];
          const TFloat x_i = positions[particle_offset + k];

          const TFloat v_i = velocities[particle_offset + k];
          // Compute new velocity
          velocities[particle_offset + k] =
            inertia_factor * v_i + cognitive_factor * c_rnd * (p_i - x_i) +
            social_factor * s_rnd * (p_g - x_i);
        }
      }
    }
  }
};

/**
 * @brief Canonical position update operator.
 *
 * This class implements the canonical position update operator, which updates
 * the position of each particle based on its current position and speed.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct CanonicalPositionUpdate : UpdatePositionFunctor<TFloat>
{

  /**
   * @brief Get the number of pseudo-random numbers required by the operator.
   *
   * @param solver Particle swarm optimization solver.
   * @return Number of pseudo-random numbers required.
   */
  uint32_t required_prns(pso_solver_cpu<TFloat>* solver) { return 0; }

  /**
   * @brief Apply the position update operator.
   *
   * @param solver Particle swarm optimization solver.
   */
  void operator()(pso_solver_cpu<TFloat>* solver)
  {
    const uint32_t ISLES = solver->_ISLES;
    const uint32_t AGENTS = solver->_AGENTS;
    const uint32_t DIMENSIONS = solver->_DIMENSIONS;

    const TFloat* velocities = const_cast<TFloat*>(solver->_velocity_vector);
    const TFloat* positions =
      const_cast<TFloat*>(solver->_population->_data_array);

    TFloat* new_positions = solver->_population->_transformed_data_array;

#pragma omp parallel for collapse(2)
    for (uint32_t i = 0; i < ISLES; i++) {
      for (uint32_t j = 0; j < AGENTS; j++) {
        const uint32_t genome_offset = i * AGENTS * DIMENSIONS + j * DIMENSIONS;
#pragma omp simd
        for (uint32_t k = 0; k < DIMENSIONS; k++) {
          const uint32_t curr_gene = genome_offset + k;

          const TFloat curr_velocity = velocities[curr_gene];
          const TFloat curr_position = positions[curr_gene];

          new_positions[curr_gene] = curr_position + curr_velocity;
        }
      }
    }
  }
};
}

#endif
