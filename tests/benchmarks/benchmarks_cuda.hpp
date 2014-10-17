#ifndef _BENCHMARKS_CUDA_H_
#define _BENCHMARKS_CUDA_H_

#include "evaluator/evaluator_cuda.hpp"
#include "evaluator/evaluable_function_cuda.hpp"

namespace locusta {

    template<typename TFloat>
    void hyper_sphere_dispatch
    (const uint32_t ISLES,
     const uint32_t AGENTS,
     const uint32_t DIMENSIONS,
     const TFloat * UPPER_BOUNDS,
     const TFloat * LOWER_BOUNDS,
     BoundMapKind bound_mapping_method,
     const bool f_negate,
     const TFloat * data,
     TFloat * evaluation_array);

    template<typename TFloat>
    struct BenchmarkCudaFunctor : EvaluationCudaFunctor<TFloat> {
        virtual void operator()(evolutionary_solver<TFloat> * solver)
            {
                const evolutionary_solver_cuda<TFloat> * _dev_solver = static_cast<evolutionary_solver_cuda<TFloat> *>(solver);

                const uint32_t ISLES = _dev_solver->_ISLES;
                const uint32_t AGENTS = _dev_solver->_AGENTS;
                const uint32_t DIMENSIONS = _dev_solver->_DIMENSIONS;
                const TFloat * const UPPER_BOUNDS = const_cast<TFloat *>(_dev_solver->_DEV_UPPER_BOUNDS);
                const TFloat * const LOWER_BOUNDS = const_cast<TFloat *>(_dev_solver->_DEV_LOWER_BOUNDS);
                const BoundMapKind bound_mapping_method = _dev_solver->_evaluator->_bound_mapping_method;
                const bool f_negate = _dev_solver->_evaluator->_f_negate;
                const TFloat * const data = const_cast<TFloat *> (_dev_solver->_dev_population->_dev_data_array);
                TFloat * const evaluation_array = _dev_solver->_dev_population->_dev_fitness_array;

                hyper_sphere_dispatch(ISLES,
                                      AGENTS,
                                      DIMENSIONS,
                                      UPPER_BOUNDS,
                                      LOWER_BOUNDS,
                                      bound_mapping_method,
                                      f_negate,
                                      data,
                                      evaluation_array);
            }
    };

} // namespace locusta
#endif /* _BENCHMARKS_CUDA_H_ */
