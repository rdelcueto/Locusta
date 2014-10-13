#ifndef _BENCHMARKS_CPU_H_
#define _BENCHMARKS_CPU_H_

#include "evaluator/evaluator.hpp"

namespace locusta {

    template<typename TFloat>
    struct BenchmarkFunctor : EvaluationFunctor<TFloat> {
        virtual void operator()(evolutionary_solver<TFloat> * solver)
            {
                const TFloat * const data = const_cast<TFloat *> (solver->_population->_data_array);
                TFloat * const evaluation_array = solver->_population->_fitness_array;

                const TFloat * const UPPER_BOUNDS = const_cast<TFloat *>(solver->_UPPER_BOUNDS);
                const TFloat * const LOWER_BOUNDS = const_cast<TFloat *>(solver->_LOWER_BOUNDS);

                const uint32_t ISLES = solver->_ISLES;
                const uint32_t AGENTS = solver->_AGENTS;
                const uint32_t DIMENSIONS = solver->_DIMENSIONS;

                const bool f_negate = solver->_evaluator->_f_negate;
                const BoundMapKind bound_mapping_method = solver->_evaluator->_bound_mapping_method;

               hyper_sphere(data,
                            evaluation_array,
                            ISLES,
                            AGENTS,
                            DIMENSIONS,
                            UPPER_BOUNDS,
                            LOWER_BOUNDS,
                            bound_mapping_method,
                            f_negate);
            }

        void hyper_sphere(const TFloat * const data,
                          TFloat * const evaluation_array,
                          const uint32_t ISLES,
                          const uint32_t AGENTS,
                          const uint32_t DIMENSIONS,
                          const TFloat * const UPPER_BOUNDS,
                          const TFloat * const LOWER_BOUNDS,
                          BoundMapKind bound_mapping_method,
                          const bool f_negate)
            {
                const uint32_t REPETITIONS = 1e2;

                for (uint32_t i = 0; i < ISLES; ++i)
                {
                    for (uint32_t j = 0; j < AGENTS; ++j)
                    {
                        const uint32_t isle = i;
                        const uint32_t agent = j;
                        TFloat reduction_sum;
                        for(uint32_t r = 0; r < REPETITIONS; ++r)
                        {
                            reduction_sum = 0.0;
                            for (uint32_t k = 0; k < DIMENSIONS; ++k)
                            {
                                TFloat x = data[isle * AGENTS * DIMENSIONS +
                                                agent * DIMENSIONS +
                                                k];
                                const TFloat &u = UPPER_BOUNDS[k];
                                const TFloat &l = LOWER_BOUNDS[k];

                                evaluator<TFloat>::bound_map(bound_mapping_method, u, l, x);
                                reduction_sum += x * x;
                            }
                        }
                        evaluation_array[isle * AGENTS + agent] = f_negate ?
                            -reduction_sum :
                            reduction_sum;
                    }
                }
            }
    };

} // namespace locusta
#endif /* _BENCHMARKS_CPU_H_ */
