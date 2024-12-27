#ifndef LOCUSTA_GA_OPERATORS_H
#define LOCUSTA_GA_OPERATORS_H

#include <inttypes.h>

namespace locusta {

/**
 * @brief Forward declaration of the ga_solver_cpu type.
 *
 * The ga_solver_cpu class is a specific implementation of the
 * evolutionary_solver for the CPU architecture, using the genetic algorithm.
 */
template<typename TFloat>
struct ga_solver_cpu;

/**
 * @brief Abstract base class for breeding operators.
 *
 * This class defines the interface for breeding operators, which are used to
 * create offspring from a population of parents.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct GaBreedFunctor
{
  /**
   * @brief Get the number of pseudo-random numbers required by the operator.
   *
   * @param solver Genetic algorithm solver.
   * @return Number of pseudo-random numbers required.
   */
  virtual uint32_t required_prns(ga_solver_cpu<TFloat>* solver) = 0;
  /**
   * @brief Apply the breeding operator.
   *
   * @param solver Genetic algorithm solver.
   */
  virtual void operator()(ga_solver_cpu<TFloat>* solver) = 0;
};

  /**
   * @brief Abstract base class for selection operators.
   * 
   * This class defines the interface for selection operators in genetic algorithms, which are used to select parents from a population.
   * 
   * @tparam TFloat Floating-point type.
   */
template<typename TFloat>
struct GaSelectionFunctor
{
  /**
   * @brief Get the number of pseudo-random numbers required by the operator.
   *
   * @param solver Genetic algorithm solver.
   * @return Number of pseudo-random numbers required.
   */
  virtual uint32_t required_prns(ga_solver_cpu<TFloat>* solver) = 0;
  /**
   * @brief Apply the selection operator.
   *
   * @param solver Genetic algorithm solver.
   */
  virtual void operator()(ga_solver_cpu<TFloat>* solver) = 0;
};
}

#endif
