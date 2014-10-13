#ifndef _BENCHMARKS_CUDA_H_
#define _BENCHMARKS_CUDA_H_

#include "evaluator/evaluator_cuda.hpp"
#include "evaluator/evaluable_function_cuda.hpp"

namespace locusta {

    template<typename TFloat>
    struct BenchmarkFunctorCuda : EvaluationCudaFunctor<TFloat> {
        virtual void operator()(evolutionary_solver<TFloat> * solver)
            {
                std::cout << "TEST\n";
            }
    };

} // namespace locusta
#endif /* _BENCHMARKS_CUDA_H_ */
