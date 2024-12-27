#ifndef LOCUSTA_PSO_OPERATORS_H
#define LOCUSTA_PSO_OPERATORS_H

#include <inttypes.h>

namespace locusta {

/**
 * @brief Forward declaration of the pso_solver_cpu type.
 *
 * The pso_solver_cpu class is a specific implementation of the
 * evolutionary_solver for the CPU architecture, using the particle swarm
 * optimization algorithm.
 */

template<typename TFloat>
struct pso_solver_cpu;

/**
 * @brief Abstract base class for updating the speed of particles in particle
 * swarm optimization.
 *
 * This class defines the interface for updating the speed of particles in
 * particle swarm optimization.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct UpdateSpeedFunctor
{
  /**
   * @brief Get the number of pseudo-random numbers required by the operator.
   *
   * @param solver Particle swarm optimization solver.
   * @return Number of pseudo-random numbers required.
   */
  virtual uint32_t required_prns(pso_solver_cpu<TFloat>* solver) = 0;

  /**
   * @brief Apply the speed update operator.
   *
   * @param solver Particle swarm optimization solver.
   */
  virtual void operator()(pso_solver_cpu<TFloat>* solver) = 0;
};

/**
 * @brief Abstract base class for updating the particle record in particle swarm
 * optimization.
 *
 * This class defines the interface for updating the particle record in particle
 * swarm optimization.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct UpdateParticleRecordFunctor
{
  /**
   * @brief Get the number of pseudo-random numbers required by the operator.
   *
   * @param solver Particle swarm optimization solver.
   * @return Number of pseudo-random numbers required.
   */
  virtual uint32_t required_prns(pso_solver_cpu<TFloat>* solver) = 0;
  /**
   * @brief Apply the particle record update operator.
   *
   * @param solver Particle swarm optimization solver.
   */
  virtual void operator()(pso_solver_cpu<TFloat>* solver) = 0;
};

/**
 * @brief Abstract base class for updating the position of particles in particle
 * swarm optimization.
 *
 * This class defines the interface for updating the position of particles in
 * particle swarm optimization.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct UpdatePositionFunctor
{
  /**
   * @brief Get the number of pseudo-random numbers required by the operator.
   *
   * @param solver Particle swarm optimization solver.
   * @return Number of pseudo-random numbers required.
   */
  virtual uint32_t required_prns(pso_solver_cpu<TFloat>* solver) = 0;
  /**
   * @brief Apply the position update operator.
   *
   * @param solver Particle swarm optimization solver.
   */
  virtual void operator()(pso_solver_cpu<TFloat>* solver) = 0;
};
}

#endif
