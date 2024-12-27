#ifndef LOCUSTA_PSO_OPERATORS_CUDA_H
#define LOCUSTA_PSO_OPERATORS_CUDA_H

#include <inttypes.h>

namespace locusta {

/**
 * @brief Forward declaration of the pso_solver_cuda type.
 *
 * The pso_solver_cuda class is a specific implementation of the
 * evolutionary_solver for the CUDA architecture, using the particle swarm
 * optimization algorithm.
 */
template<typename TFloat>
struct pso_solver_cuda;

/**
 * @brief Abstract base class for updating the speed of particles in particle
 * swarm optimization on CUDA.
 *
 * This class defines the interface for updating the speed of particles in
 * particle swarm optimization on CUDA.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct UpdateSpeedCudaFunctor
{
  /**
   * @brief Get the number of pseudo-random numbers required by the operator.
   *
   * @param solver Particle swarm optimization solver.
   * @return Number of pseudo-random numbers required.
   */
  virtual uint32_t required_prns(pso_solver_cuda<TFloat>* solver) = 0;
  /**
   * @brief Apply the speed update operator.
   *
   * @param solver Particle swarm optimization solver.
   */
  virtual void operator()(pso_solver_cuda<TFloat>* solver) = 0;
};

/**
 * @brief Abstract base class for updating the particle record in particle swarm
 * optimization on CUDA.
 *
 * This class defines the interface for updating the particle record in particle
 * swarm optimization on CUDA.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct UpdateParticleRecordCudaFunctor
{
  /**
   * @brief Get the number of pseudo-random numbers required by the operator.
   *
   * @param solver Particle swarm optimization solver.
   * @return Number of pseudo-random numbers required.
   */
  virtual uint32_t required_prns(pso_solver_cuda<TFloat>* solver) = 0;
  /**
   * @brief Apply the particle record update operator.
   *
   * @param solver Particle swarm optimization solver.
   */
  virtual void operator()(pso_solver_cuda<TFloat>* solver) = 0;
};

/**
 * @brief Abstract base class for updating the position of particles in particle
 * swarm optimization on CUDA.
 *
 * This class defines the interface for updating the position of particles in
 * particle swarm optimization on CUDA.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct UpdatePositionCudaFunctor
{
  /**
   * @brief Get the number of pseudo-random numbers required by the operator.
   *
   * @param solver Particle swarm optimization solver.
   * @return Number of pseudo-random numbers required.
   */
  virtual uint32_t required_prns(pso_solver_cuda<TFloat>* solver) = 0;
  /**
   * @brief Apply the position update operator.
   *
   * @param solver Particle swarm optimization solver.
   */
  virtual void operator()(pso_solver_cuda<TFloat>* solver) = 0;
};
}

#endif
