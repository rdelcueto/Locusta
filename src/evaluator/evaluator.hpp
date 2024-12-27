#ifndef LOCUSTA_EVALUATOR_H
#define LOCUSTA_EVALUATOR_H

#include <functional>
#include <stdint.h>

#include "evaluable_function.hpp"

namespace locusta {

/**
 * @brief Forward declaration of the evolutionary_solver type.
 * See locusta::evolutionary_solver for the full definition.
 *
 * The evolutionary_solver class composes the locusta defined classes:
 * population_set, evaluator, and prngenerator. A solver setup will contain
 * instances of these classes based on the specific type of solver, implementing
 * the corresponding meta-heuristic.
 */
template<typename TFloat>
struct evolutionary_solver;

/**
 * @brief Enum defining the different types of bound mapping methods.
 *
 * Each evaluator must define the fitness function domain constraints. Values
 * outside this domain must be remapped to well-defined values. This enum
 * defines the type of mapping used by the evaluator instance.
 *
 * Locusta currently implements the following methods:
 *  - IgnoreBounds: No mapping is applied.
 *  - CropBounds: Values outside the domain are cropped to the nearest bound.
 *  - MirrorBounds: Values outside the domain are mirrored around the nearest
 * bound.
 *
 * Note that a user of the Locusta library can implement their own evaluator
 * class, extending the BoundMapKind and implementing their own bound mapping
 * function.
 */
enum BoundMapKind
{
  IgnoreBounds,
  CropBounds,
  MirrorBounds
};

/**
 * @brief Abstract evaluator type.
 *
 * The evaluator abstract type cannot be instantiated, but can be used as a base
 * class specific to the target architecture: CPU or CUDA. This class is
 * responsible for evaluating the fitness of a population of candidate
 * solutions.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct evaluator
{
  /**
   * @brief Construct a new evaluator object.
   *
   * @param eval_functor Evaluation functor. This is a pointer to the fitness
   * function implementation. In the Locusta library design, there is no
   * concrete implementation for this function, as it will be defined by the
   * library user when setting up a solver and the corresponding evaluator class
   * instantiation for solving a particular optimization problem. The purpose of
   * a functor "function pointer" is to allow flexibility for the user to define
   * their own fitness function. Also, CPU and CUDA implementation have
   * different data schemes appropriate for efficient computation in each
   * computer architecture. Using function pointers allows to abstract those
   * details between implementations. For reference, the Locusta library
   * includes a set of usage examples and concrete implementation of the
   * _evaluation_functor. See BenchmarkFunctor.
   * @param f_negate Flag describing whether the evaluator will use the fitness
   * value or its negative. When solving optimization problems, the user might
   * want to maximize or minimize the fitness function. This flag defines
   * whether the fitness value should be negated, effectively switching between
   * maximizing or minimizing the fitness function in the solver.
   * @param bound_mapping_method Specified the bounding map method.
   * @param prn_numbers Number of pseudo random numbers needed per evaluation.
   * To store the pseudo random numbers required for the evaluation of the given
   * fitness function.
   */
  evaluator(EvaluationFunctor<TFloat>* eval_functor,
            bool f_negate,
            BoundMapKind bound_mapping_method,
            uint32_t prn_numbers)
    : _evaluation_functor(eval_functor)
    , _f_negate(f_negate)
    , _bound_mapping_method(bound_mapping_method)
    , _eval_prn_size(prn_numbers)
  {
  }

  /**
   * @brief Destroy the evaluator object.
   */
  virtual ~evaluator() {}

  /**
   *  @brief Evaluate the solver's population data set.
   *
   * See locusta::evolutionary_solver::advance() for more details.
   *  @param solver A pointer to the solver object.
   */
  /**
   * @brief Evaluate the solver's population data set.
   *
   * This API is call by the solver's advance() method, to evaluate the
   * population_set's genomes array (_data_array) using the user-defined
   * _evaluation_functor to populate the resulting values in the
   * population_set's genome fitness array (_fitness_array).
   *
   * @param solver Evolutionary solver.
   */
  virtual void evaluate(evolutionary_solver<TFloat>* solver) = 0;

  /**
   * @brief Bound mapping implementation.
   *
   * This API is called by the solver's transform() function, to execute the
   * bound mapping method on the population_set's transformed_data_array. See
   * locusta::evolutionary_solver::transform() for more details.
   *
   * @param bound_mapping_method Bound mapping method.
   * @param u Upper bound.
   * @param l Lower bound.
   * @param x Value to map.
   */
  virtual void bound_map(BoundMapKind bound_mapping_method,
                         const TFloat& u,
                         const TFloat& l,
                         TFloat& x) = 0;

  /**
   * @brief Flag describing whether the evaluator will use the fitness value or
   * its negative.
   */
  const bool _f_negate;

  /**
   * @brief Specified the bounding map method.
   */
  const BoundMapKind _bound_mapping_method;

  /**
   * @brief Number of pseudo random numbers needed per evaluation.
   */
  const size_t _eval_prn_size;

  /**
   * @brief Evaluation random numbers array.
   */
  TFloat* _eval_prn_numbers;

  /**
   * @brief Evaluation dispatch functor.
   */
  EvaluationFunctor<TFloat>* _evaluation_functor;

  /// @see locusta::EvaluationFunctor
};

} // namespace locusta

#endif
