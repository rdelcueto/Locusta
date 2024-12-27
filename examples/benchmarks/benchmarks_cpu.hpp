#ifndef _BENCHMARKS_CPU_H_
#define _BENCHMARKS_CPU_H_

#include "evaluator/evaluator_cpu.hpp"
#include "prngenerator/prngenerator_cpu.hpp"

namespace locusta {

/**
 * @brief Dispatch function for benchmark functions.
 *
 * This function dispatches the evaluation of a benchmark function to the
 * appropriate implementation.
 *
 * @param ISLES Number of isles in the population.
 * @param AGENTS Number of agents per isle.
 * @param DIMENSIONS Number of dimensions per agent.
 * @param F_NEGATE_EVALUATION Flag indicating whether to negate the evaluation
 * result.
 * @param FUNC_ID Identifier of the benchmark function to evaluate.
 * @param EVALUATION_BIAS Bias value to add to the evaluation result.
 * @param SHIFT_ORIGIN Array of shift values to apply to the input data.
 * @param F_ROTATE Flag indicating whether to apply a rotation to the input
 * data.
 * @param ROTATION_MATRIX Matrix of rotation values to apply to the input data.
 * @param evaluation_data Input data to evaluate.
 * @param evaluation_results Output array to store the evaluation results.
 * @param local_generator Pseudo-random number generator.
 */
template<typename TFloat>
void
benchmark_dispatch(const uint32_t ISLES,
                   const uint32_t AGENTS,
                   const uint32_t DIMENSIONS,
                   const bool F_NEGATE_EVALUATION,
                   const uint32_t FUNC_ID,
                   const TFloat EVALUATION_BIAS,
                   const TFloat* SHIFT_ORIGIN,
                   const bool F_ROTATE,
                   const TFloat* ROTATION_MATRIX,
                   const TFloat* evaluation_data,
                   TFloat* evaluation_results,
                   prngenerator_cpu<TFloat>* local_generator);

/**
 * @brief Evaluation functor for benchmark functions.
 *
 * This class implements the EvaluationFunctor interface for benchmark
 * functions.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct BenchmarkFunctor : EvaluationFunctor<TFloat>
{
  /**
   * @brief Enum defining the identifiers of the available benchmark functions.
   */
  enum FUNCTION_IDENTIFIERS
  {
    SPHERE = 1,      // 1
    ROT_ELLIPS,      // 2
    ROT_BENT_CIGAR,  // 3
    ROT_DISCUS,      // 4
    DIFF_POWERS,     // 5
    ROT_ROSENBROCK,  // 6
    ROT_SCHAFFER,    // 7
    ROT_ACKLEY,      // 8
    ROT_WEIERSTRASS, // 9
    ROT_GRIEWANK,    // 10
    RASTRIGIN,       // 11
    ROT_RASTRIGIN    // 12
  };

  const uint32_t _FUNCTION_ID;
  const uint32_t _DIMENSIONS;

  TFloat _EVALUATION_BIAS;
  TFloat* _SHIFT_ORIGIN;
  TFloat* _ROTATION_MATRIX;
  uint32_t _ROT_FLAG;

  /**
   * @brief Construct a new BenchmarkFunctor object.
   *
   * @param function_id Identifier of the benchmark function to evaluate.
   * @param dimensions Number of dimensions per agent.
   */
  BenchmarkFunctor(uint32_t function_id, uint32_t dimensions)
    : EvaluationFunctor<TFloat>()
    , _FUNCTION_ID(function_id)
    , _DIMENSIONS(dimensions)
  {

    // TODO: Load CEC transformation matrices/vectors.

    switch (function_id) {
      case ROT_ELLIPS:
        _EVALUATION_BIAS = -1300.0;
        _ROT_FLAG = 1;
        break;
      case ROT_BENT_CIGAR:
        _EVALUATION_BIAS = -1200.0;
        _ROT_FLAG = 1;
        break;
      case ROT_DISCUS:
        _EVALUATION_BIAS = -1100.0;
        _ROT_FLAG = 1;
        break;
      case DIFF_POWERS:
        _EVALUATION_BIAS = -1000.0;
        break;
      default: // SPHERE DEFAULT FUNC
        _EVALUATION_BIAS = -1400.0;
        break;
    }

    _SHIFT_ORIGIN = new TFloat[_DIMENSIONS];
    _ROTATION_MATRIX = new TFloat[_DIMENSIONS * _DIMENSIONS];
  }
  /**
   * @brief Destroy the BenchmarkFunctor object.
   */
  ~BenchmarkFunctor()
  {
    delete[] _SHIFT_ORIGIN;
    delete[] _ROTATION_MATRIX;
  }

  /**
   * @brief Function call operator.
   *
   * This operator evaluates the fitness of a population of candidate solutions
   * using a benchmark function.
   *
   * @param solver Evolutionary solver.
   */
  virtual void operator()(evolutionary_solver<TFloat>* solver)
  {
    const uint32_t ISLES = solver->_ISLES;
    const uint32_t AGENTS = solver->_AGENTS;
    const uint32_t DIMENSIONS = solver->_DIMENSIONS;
    const bool F_NEGATE_EVALUATION = solver->_evaluator->_f_negate;
    const TFloat* evaluation_data =
      const_cast<TFloat*>(solver->_population->_data_array);
    TFloat* evaluation_results = solver->_population->_fitness_array;
    prngenerator_cpu<TFloat>* local_generator =
      static_cast<prngenerator_cpu<TFloat>*>(solver->_bulk_prn_generator);

    benchmark_dispatch(ISLES,
                       AGENTS,
                       DIMENSIONS,
                       F_NEGATE_EVALUATION,
                       _FUNCTION_ID,
                       _EVALUATION_BIAS,
                       _SHIFT_ORIGIN,
                       _ROT_FLAG,
                       _ROTATION_MATRIX,
                       evaluation_data,
                       evaluation_results,
                       local_generator);
  }
};

} // namespace locusta

#include "benchmarks_cpu_impl.hpp"
#endif /* _BENCHMARKS_CPU_H_ */
