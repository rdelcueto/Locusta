#include "evolutionary_solver_cpu.hpp"

namespace locusta {

    template<typename TFloat>
    evolutionary_solver_cpu<TFloat>::evolutionary_solver_cpu(population_set_cpu<TFloat> * population,
                                                               evaluator_cpu<TFloat> * evaluator,
                                                               prngenerator_cpu<TFloat> * prn_generator,
                                                               uint32_t generation_target,
                                                               TFloat * upper_bounds,
                                                               TFloat * lower_bounds)
    : evolutionary_solver<TFloat>(population,
                                  evaluator,
                                  prn_generator,
                                  generation_target,
                                  upper_bounds,
                                  lower_bounds)
    {
    }

    template<typename TFloat>
    evolutionary_solver_cpu<TFloat>::~evolutionary_solver_cpu()
    {
    }

} // namespace locusta
