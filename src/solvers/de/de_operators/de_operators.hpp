#ifndef LOCUSTA_DE_OPERATORS_H
#define LOCUSTA_DE_OPERATORS_H

#include <inttypes.h>

namespace locusta {

/**
 * @brief Forward declaration of the de_solver_cpu type.
 *
 * The de_solver_cpu class is a specific implementation of the
 * evolutionary_solver for the CPU architecture, using the differential
 * evolution algorithm.
 */
template<typename TFloat>
struct de_solver_cpu;

/**
 * @brief Abstract base class for differential evolution breeding operators.
 *
 * This class defines the interface for differential evolution breeding
 * operators, which are used to create offspring from a population of parents.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct DeBreedFunctor
{
  /**
   * @brief Get the number of pseudo-random numbers required by the operator.
   *
   * @param solver Differential evolution solver.
   * @return Number of pseudo-random numbers required.
   */
  virtual uint32_t required_prns(de_solver_cpu<TFloat>* solver) = 0;
  /**
   * @brief Apply the breeding operator.
   *
   * @param solver Differential evolution solver.
   */
  virtual void operator()(de_solver_cpu<TFloat>* solver) = 0;
};

/**
 * @brief Abstract base class for differential evolution selection operators.
 *
 * This class defines the interface for differential evolution selection
 * operators, which are used to select parents from a population.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct DeSelectionFunctor
{
  /**
   * @brief Get the number of pseudo-random numbers required by the operator.
   *
   * @param solver Differential evolution solver.
   * @return Number of pseudo-random numbers required.
   */
  virtual uint32_t required_prns(de_solver_cpu<TFloat>* solver) = 0;
  /**
   * @brief Apply the selection operator.
   *
   * @param solver Differential evolution solver.
   */
  virtual void operator()(de_solver_cpu<TFloat>* solver) = 0;
};
}

#endif
