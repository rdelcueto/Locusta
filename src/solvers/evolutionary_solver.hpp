#ifndef LOCUSTA_EVOLUTIONARY_SOLVER_H
#define LOCUSTA_EVOLUTIONARY_SOLVER_H

#include <limits>

#include "../prngenerator/prngenerator.hpp"
#include "../population/population_set.hpp"

namespace locusta {

  ///Interface for evolutionary computing metaheuristic solvers
  template<typename TFloat>
  class evolutionary_solver {

  public:

    evolutionary_solver(population_set<TFloat> * population,
                        evaluator<TFloat> * evaluator,
                        prngenerator<TFloat> * prn_generator)
      : _population(population),
        _evaluator(evaluator),
        _bulk_prn_generator(prn_generator),
        _f_initialized(false)
    {}

    virtual ~evolutionary_solver() {}

    /// Prints the current solver's configuration.
    virtual void _print_solver_config() = 0;

    /// Prints the current elite set
    virtual void _print_solver_elite() = 0;

    /// Prints the current best solution.
    virtual void _print_solver_solution() = 0;

    /// Initializes and allocated solver's dynamic resources.
    virtual void _initialize() = 0;

    /// Finalizes and frees solver's dynamic resources.
    virtual void _finalize() = 0;

  protected:

    /// Describes the state of the solver.
    bool _f_initialized;

    /// Population Set
    population_set<TFloat> * const _population;

    /// Evaluator
    evaluator<TFloat> * const _evaluator;

    /// Bulk Pseudo Random Number Generator
    prngenerator<TFloat> * const _bulk_prn_generator;

    /// Bulk Pseudo Random Number array
    TFloat * _bulk_prnumbers;

    /// Describes the size of the _bulk_prnumbers array.
    size_t _bulk_size;

    ///Counter describing the solver's current generation.
    size_t _generation_count;

    ///Counter describint the solver's target generation count.
    size_t _generation_target;

  };

} // namespace locusta
#endif
