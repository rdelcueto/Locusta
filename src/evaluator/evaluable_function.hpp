#ifndef LOCUSTA_EVALUABLE_FUNCTION_H
#define LOCUSTA_EVALUABLE_FUNCTION_H

#include <limits>

namespace locusta {
/**
 * @brief Forward declaration of the evolutionary_solver type.
 *
 * The evolutionary_solver class composes the locusta defined classes:
 * population_set, evaluator, and prngenerator. A solver setup will contain
 * instances of these classes based on the specific type of solver, implementing
 * the corresponding meta-heuristic.
 */
template<typename TFloat>
struct evolutionary_solver;
/**
 * @brief Abstract base class for evaluation functors.
 *
 * This class defines the interface for evaluation functors, which are used to
 * evaluate the fitness of a population of candidate solutions.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct EvaluationFunctor
{
  /**
   * @brief Function call operator.
   *
   * This operator is used to evaluate the fitness of a population of candidate
   * solutions.
   *
   * @param solver Evolutionary solver.
   */
  virtual void operator()(evolutionary_solver<TFloat>* solver) {};
};
}

#endif
