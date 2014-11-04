#ifndef _BENCHMARKS_CPU_H_
#define _BENCHMARKS_CPU_H_

#include "evaluator/evaluator_cpu.hpp"

#include "prngenerator/prngenerator_cpu.hpp"

namespace locusta {

    template<typename TFloat>
    inline TFloat sphere(const uint32_t DIMENSIONS,
                  const uint32_t DIMENSION_OFFSET,
                  const TFloat * evaluation_vector) {

        TFloat reduction_sum = 0.0;

        for (uint32_t k = 0; k < DIMENSIONS; ++k) {
            const TFloat x = evaluation_vector[k * DIMENSION_OFFSET];
            reduction_sum += x * x;
        }

        return reduction_sum;
    }

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
     prngenerator_cpu<TFloat> * local_generator) {

        const uint32_t REPETITIONS = 1e2;

        for(uint32_t i = 0; i < ISLES; ++i) {
            for(uint32_t j = 0; j < AGENTS; ++j) {
                const uint32_t isle = i;
                const uint32_t agent = j;

                TFloat result = 0;
                const TFloat * genome = evaluation_data + i * AGENTS * DIMENSIONS + j * DIMENSIONS;
                for(uint32_t r = 0; r < REPETITIONS; ++r) {
                    switch (FUNC_ID) {
                    default:
                        result = sphere(DIMENSIONS,
                                        1,
                                        genome);
                        break;
                    }
                }

                evaluation_results[isle * AGENTS + agent] = F_NEGATE_EVALUATION ?
                    -result : result;
            }
        }
    }

    template<typename TFloat>
    struct BenchmarkFunctor : EvaluationFunctor<TFloat> {

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
        TFloat * _ROTATION_MATRIX;
        uint32_t _ROT_FLAG;

        BenchmarkFunctor(uint32_t function_id, uint32_t dimensions) : EvaluationFunctor<TFloat>(), _FUNCTION_ID(function_id),
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
            _ROTATION_MATRIX = new TFloat[_DIMENSIONS * _DIMENSIONS];

        }

        ~BenchmarkFunctor() {

        }

        virtual void operator()(evolutionary_solver<TFloat> * solver)
            {
                const uint32_t ISLES = solver->_ISLES;
                const uint32_t AGENTS = solver->_AGENTS;
                const uint32_t DIMENSIONS = solver->_DIMENSIONS;
                const bool F_NEGATE_EVALUATION = solver->_evaluator->_f_negate;
                const TFloat * evaluation_data = const_cast<TFloat *> (solver->_population->_data_array);
                TFloat * evaluation_results = solver->_population->_fitness_array;
                prngenerator_cpu<TFloat> * local_generator = static_cast<prngenerator_cpu<TFloat>*>(solver->_bulk_prn_generator);

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
#endif /* _BENCHMARKS_CPU_H_ */
