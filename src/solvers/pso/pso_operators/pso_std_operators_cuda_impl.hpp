#ifndef LOCUSTA_PSO_STD_OPERATORS_CUDA_H
#define LOCUSTA_PSO_STD_OPERATORS_CUDA_H

#include "pso_operators_cuda.hpp"

namespace locusta {

template <typename TFloat>
void canonical_particle_update_dispatch(
  const uint32_t ISLES, const uint32_t AGENTS, const uint32_t DIMENSIONS,
  const TFloat* positions, const TFloat* fitness, TFloat* record_positions,
  TFloat* record_fitness);

template <typename TFloat>
void canonical_speed_update_dispatch(
  const uint32_t ISLES, const uint32_t AGENTS, const uint32_t DIMENSIONS,
  const TFloat inertia_factor, const TFloat cognitive_factor,
  const TFloat social_factor, const TFloat* positions,
  const TFloat* record_positions, const TFloat* isle_record_positions,
  const TFloat* prng_vector, TFloat* velocities);

template <typename TFloat>
void canonical_position_update_dispatch(
  const uint32_t ISLES, const uint32_t AGENTS, const uint32_t DIMENSIONS,
  const TFloat* velocities, const TFloat* positions, TFloat* new_positions);

template <typename TFloat>
struct CanonicalParticleRecordUpdateCuda
  : UpdateParticleRecordCudaFunctor<TFloat>
{

  virtual uint32_t required_prns(pso_solver_cuda<TFloat>* solver) { return 0; }

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

    canonical_particle_update_dispatch(ISLES, AGENTS, DIMENSIONS, positions,
                                       fitness, record_positions,
                                       record_fitness);
  }
};

template <typename TFloat>
struct CanonicalSpeedUpdateCuda : UpdateSpeedCudaFunctor<TFloat>
{

  virtual uint32_t required_prns(pso_solver_cuda<TFloat>* solver)
  {
    const uint32_t ISLES = solver->_ISLES;
    const uint32_t AGENTS = solver->_AGENTS;
    const uint32_t DIMENSIONS = solver->_DIMENSIONS;

    return ISLES * AGENTS * DIMENSIONS * 2;
  }

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

    canonical_speed_update_dispatch(ISLES, AGENTS, DIMENSIONS, inertia_factor,
                                    cognitive_factor, social_factor, positions,
                                    record_positions, isle_record_positions,
                                    prng_vector, velocities);
  }
};

template <typename TFloat>
struct CanonicalPositionUpdateCuda : UpdatePositionCudaFunctor<TFloat>
{

  virtual uint32_t required_prns(pso_solver_cuda<TFloat>* solver) { return 0; }

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

    canonical_position_update_dispatch(ISLES, AGENTS, DIMENSIONS, velocities,
                                       positions, new_positions);
  }
};
}

#endif
