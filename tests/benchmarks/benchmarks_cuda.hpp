#ifndef _BENCHMARKS_CUDA_H_
#define _BENCHMARKS_CUDA_H_

#include "evaluator/evaluator_cuda.hpp"
#include "evaluator/evaluable_function_cuda.hpp"

#include "prngenerator/prngenerator_cuda.hpp"

namespace locusta {

    template<typename TFloat>
    void benchmark_dispatch
    (const uint32_t ISLES,
     const uint32_t AGENTS,
     const uint32_t DIMENSIONS,
     const bool F_NEGATE_EVALUATION,
     const uint32_t FUNC_ID,
     const TFloat EVALUATION_BIAS,
     const TFloat * SHIFT_ORIGIN,
     const bool F_ROTATE,
     const TFloat * ROTATION_MATRIX,
     const TFloat * evaluation_data,
     TFloat * evaluation_results,
     prngenerator_cuda<TFloat> * local_generator);

    template<typename TFloat>
    struct BenchmarkCudaFunctor : EvaluationCudaFunctor<TFloat> {

        enum FUNCTION_IDENTIFIERS {
            SPHERE = 1,
            ROT_ELLIPS,
            ROT_BENT_CIGAR,
            ROT_DISCUS,
            DIFF_POWERS,
            ROT_ROSENBROCK,
            ROT_SCHAFFER,
            ROT_ACKLEY,
            ROT_WEIERSTRASS,
            ROT_GRIEWANK,
            RASTRIGIN,
            ROT_RASTRIGIN
        };

        const uint32_t _FUNCTION_ID;
        const uint32_t _DIMENSIONS;

        TFloat _EVALUATION_BIAS;
        TFloat * _SHIFT_ORIGIN;
        TFloat * _DEV_SHIFT_ORIGIN;
        TFloat * _ROTATION_MATRIX;
        TFloat * _DEV_ROTATION_MATRIX;
        uint32_t _ROT_FLAG;

        BenchmarkCudaFunctor(uint32_t function_id, uint32_t dimensions) : EvaluationCudaFunctor<TFloat>(), _FUNCTION_ID(function_id),
            _DIMENSIONS(dimensions){

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
            CudaSafeCall(cudaMalloc((void **) &(_DEV_SHIFT_ORIGIN), _DIMENSIONS * sizeof(TFloat)));

            _ROTATION_MATRIX = new TFloat[_DIMENSIONS * _DIMENSIONS];
            CudaSafeCall(cudaMalloc((void **) &(_DEV_ROTATION_MATRIX), _DIMENSIONS * _DIMENSIONS * sizeof(TFloat)));

        }

        ~BenchmarkCudaFunctor() {
            CudaSafeCall(cudaFree(_DEV_SHIFT_ORIGIN));
            CudaSafeCall(cudaFree(_DEV_ROTATION_MATRIX));

            delete [] _SHIFT_ORIGIN;
            delete [] _ROTATION_MATRIX;
        }

        virtual void operator()(evolutionary_solver<TFloat> * solver)
            {
                const evolutionary_solver_cuda<TFloat> * _dev_solver = static_cast<evolutionary_solver_cuda<TFloat> *>(solver);

                const uint32_t ISLES = _dev_solver->_ISLES;
                const uint32_t AGENTS = _dev_solver->_AGENTS;
                const uint32_t DIMENSIONS = _dev_solver->_DIMENSIONS;

                const bool F_NEGATE_EVALUATION = _dev_solver->_evaluator->_f_negate;
                const TFloat * const evaluation_data = const_cast<TFloat *> (_dev_solver->_dev_population->_dev_data_array);
                TFloat * const evaluation_results = _dev_solver->_dev_population->_dev_fitness_array;
                prngenerator_cuda<TFloat> * local_generator = _dev_solver->_dev_bulk_prn_generator;

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
#endif /* _BENCHMARKS_CUDA_H_ */
