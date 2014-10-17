#include "gtest/gtest.h"

#include <iostream>
#include <algorithm>
#include <time.h>

#include "./benchmarks/benchmarks_cpu.hpp"
#include "./benchmarks/benchmarks_cuda.hpp"

#include "evaluator/evaluator_cpu.hpp"
#include "evaluator/evaluator_cuda.hpp"

#include "prngenerator/prngenerator_cpu.hpp"
#include "prngenerator/prngenerator_cuda.hpp"

#include "solvers/pso/pso_solver_cpu.hpp"
#include "solvers/pso/pso_solver_cuda.hpp"

#include "solvers/pso/pso_operators/pso_std_operators_cpu_impl.hpp"
#include "solvers/pso/pso_operators/pso_std_operators_cuda_impl.hpp"

#include "cuda_runtime.h"

using namespace locusta;

class LocustaTest : public testing::Test {
protected:

    static void SetUpTestCase()
        {
            __setup_cuda();
        }

    virtual void SetUp()
        {
            // Init timer
            start_time = time(NULL);

            evaluation_functor_cpu_ptr = new BenchmarkFunctor<float>();
            evaluator_cpu_ptr = new evaluator_cpu<float>(evaluation_functor_cpu_ptr,
                                                         true,
                                                         BoundMapKind::CropBounds,
                                                         DIMENSIONS);

            evaluation_functor_cuda_ptr = new BenchmarkCudaFunctor<float>();
            evaluator_cuda_ptr = new evaluator_cuda<float>(evaluation_functor_cuda_ptr,
                                                           true,
                                                           BoundMapKind::CropBounds,
                                                           DIMENSIONS);

            prngenerator_cpu_ptr = new prngenerator_cpu<float>(ISLES * AGENTS);
            prngenerator_cpu_ptr->_initialize_engines(SEED);

            prngenerator_cuda_ptr = new prngenerator_cuda<float>(ISLES * AGENTS);
            prngenerator_cuda_ptr->_initialize_engines(SEED);

            population_cpu_ptr = new population_set_cpu<float>(ISLES, AGENTS, DIMENSIONS);
            population_cuda_ptr = new population_set_cuda<float>(ISLES, AGENTS, DIMENSIONS);

            upper_bounds_ptr = new float[DIMENSIONS];
            lower_bounds_ptr = new float[DIMENSIONS];

            // Bounds definition
            std::fill(upper_bounds_ptr, upper_bounds_ptr + DIMENSIONS, 2.0f);
            std::fill(lower_bounds_ptr, lower_bounds_ptr + DIMENSIONS, 0.0f);
        }

    virtual void TearDown()
        {
            const time_t end_time = time(NULL);
            const time_t elapsed_time = end_time - start_time;

            delete [] lower_bounds_ptr;
            delete [] upper_bounds_ptr;

            delete population_cpu_ptr;
            delete population_cuda_ptr;

            delete evaluator_cpu_ptr;
            delete evaluator_cuda_ptr;

            delete prngenerator_cpu_ptr;
            delete prngenerator_cuda_ptr;

            RecordProperty("Elapsed Time", elapsed_time);
        }

    time_t start_time;

    // Pseudo Random Number Generators
    prngenerator_cpu<float> * prngenerator_cpu_ptr;
    EvaluationFunctor<float> * evaluation_functor_cpu_ptr;

    prngenerator_cuda<float> * prngenerator_cuda_ptr;
    EvaluationCudaFunctor<float> * evaluation_functor_cuda_ptr;

    // Evaluator
    evaluator_cpu<float> * evaluator_cpu_ptr;
    evaluator_cuda<float> * evaluator_cuda_ptr;

    // Population
    const uint64_t SEED = 1;
    const uint32_t GENERATIONS = 3e1;
    const uint32_t ISLES = 1;
    const uint32_t AGENTS = 32;
    const uint32_t DIMENSIONS = 32;

    population_set_cpu<float> * population_cpu_ptr;
    population_set_cuda<float> * population_cuda_ptr;

    float * upper_bounds_ptr;
    float * lower_bounds_ptr;

};

class ParticleSwarmTest : public LocustaTest {
    virtual void SetUp()
        {
            LocustaTest::SetUp();
            pso_solver_cpu_ptr = new pso_solver_cpu<float>(population_cpu_ptr,
                                                           evaluator_cpu_ptr,
                                                           prngenerator_cpu_ptr,
                                                           GENERATIONS,
                                                           upper_bounds_ptr,
                                                           lower_bounds_ptr);

            pso_solver_cuda_ptr = new pso_solver_cuda<float>(population_cuda_ptr,
                                                             evaluator_cuda_ptr,
                                                             prngenerator_cuda_ptr,
                                                             GENERATIONS,
                                                             upper_bounds_ptr,
                                                             lower_bounds_ptr);

        }

    virtual void TearDown()
        {
            delete pso_solver_cpu_ptr;
            delete pso_solver_cuda_ptr;
            LocustaTest::TearDown();
        }

public:
    pso_solver_cpu<float> * pso_solver_cpu_ptr;
    pso_solver_cuda<float> * pso_solver_cuda_ptr;

};

TEST_F(ParticleSwarmTest, BasicTest)
{
    pso_solver_cpu_ptr->setup_solver();
    pso_solver_cpu_ptr->setup_operators(new CanonicalSpeedUpdate<float>(),
                                        new CanonicalPositionUpdate<float>());
    pso_solver_cpu_ptr->run();
    //pso_solver_cpu_ptr->print_solutions();
}

TEST_F(ParticleSwarmTest, BasicCudaTest)
{
    pso_solver_cuda_ptr->setup_solver();
    pso_solver_cuda_ptr->setup_operators(new CanonicalSpeedUpdateCuda<float>(),
                                         new CanonicalPositionUpdateCuda<float>());
    pso_solver_cuda_ptr->run();
    // pso_solver_cuda_ptr->print_population();
}
