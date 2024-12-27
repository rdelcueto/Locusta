#ifndef LOCUSTA_GA_OPERATORS_CUDA_H
#define LOCUSTA_GA_OPERATORS_CUDA_H

#include <inttypes.h>

namespace locusta {

/**
 * @brief Forward declaration of the ga_solver_cuda type.
 *
 * The ga_solver_cuda class is a specific implementation of the
 * evolutionary_solver for the CUDA architecture, using the genetic algorithm.
 */
template<typename TFloat>
struct ga_solver_cuda;

/**
 * @brief Abstract base class for breeding operators on CUDA.
 *
 * This class defines the interface for breeding operators on CUDA, which are
 * used to create offspring from a population of parents.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct BreedCudaFunctor
{
  /**
   * @brief Get the number of pseudo-random numbers required by the operator.
   *
   * @param solver Genetic algorithm solver.
   * @return Number of pseudo-random numbers required.
   */
  virtual uint32_t required_prns(ga_solver_cuda<TFloat>* solver) = 0;
  /**
   * @brief Apply the breeding operator.
   *
   * @param solver Genetic algorithm solver.
   */
  virtual void operator()(ga_solver_cuda<TFloat>* solver) = 0;
};

/**
 * @brief Abstract base class for selection operators on CUDA.
 *
 * This class defines the interface for selection operators on CUDA, which are
 * used to select parents from a population.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct SelectionCudaFunctor
{
  /**
   * @brief Get the number of pseudo-random numbers required by the operator.
   *
   * @param solver Genetic algorithm solver.
   * @return Number of pseudo-random numbers required.
   */
  virtual uint32_t required_prns(ga_solver_cuda<TFloat>* solver) = 0;
  /**
   * @brief Apply the selection operator.
   *
   * @param solver Genetic algorithm solver.
   */
  virtual void operator()(ga_solver_cuda<TFloat>* solver) = 0;
};
}

#endif
