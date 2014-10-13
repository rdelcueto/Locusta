#include "gtest/gtest.h"

#include <iostream>
#include <algorithm>
#include <time.h>

#include "./benchmarks/benchmarks_cpu.hpp"

#include "solvers/pso/pso_solver.hpp"
#include "solvers/pso/pso_operators/pso_std_operators.hpp"

#include "evaluator/evaluator.hpp"
#include "prngenerator/prngenerator_cpu.hpp"

#include "solvers/evolutionary_solver_cuda.hpp"
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

            EvaluationFunctor<float> * evaluation_functor_cpu_ptr = new BenchmarkFunctor<float>();

            evaluator_cpu_ptr = new evaluator<float>(evaluation_functor_cpu_ptr,
                                                     true,
                                                     BoundMapKind::CropBounds,
                                                     DIMENSIONS);

            prngenerator_cpu_ptr = new prngenerator_cpu<float>(ISLES * AGENTS);
            prngenerator_cpu_ptr->_initialize_engines(SEED);

            prngenerator_cuda_ptr = new prngenerator_cuda<float>(ISLES * AGENTS);
            prngenerator_cuda_ptr->_initialize_engines(SEED);

            population_cpu_ptr = new population_set<float>(ISLES, AGENTS, DIMENSIONS);
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
    prngenerator<float> * prngenerator_cpu_ptr;
    prngenerator_cuda<float> * prngenerator_cuda_ptr;

    // Evaluator
    evaluator<float> * evaluator_cpu_ptr;
    evaluator_cuda<float> * evaluator_cuda_ptr;

    // Population
    const uint64_t SEED = 1;
    const uint32_t GENERATIONS = 1e2;
    const uint32_t ISLES = 1;
    const uint32_t AGENTS = 128;
    const uint32_t DIMENSIONS = 128;

    population_set<float> * population_cpu_ptr;
    population_set_cuda<float> * population_cuda_ptr;

    float * upper_bounds_ptr;
    float * lower_bounds_ptr;

};

class ParticleSwarmTest : public LocustaTest {
    virtual void SetUp()
        {
            LocustaTest::SetUp();
            pso_solver_ptr = new pso_solver<float>(population_cpu_ptr,
                                                   evaluator_cpu_ptr,
                                                   prngenerator_cpu_ptr,
                                                   GENERATIONS,
                                                   upper_bounds_ptr,
                                                   lower_bounds_ptr);
        }

    virtual void TearDown()
        {
            delete pso_solver_ptr;
            LocustaTest::TearDown();
        }

public:
    pso_solver<float> * pso_solver_ptr;

};

TEST_F(ParticleSwarmTest, BasicTest)
{
    pso_solver_ptr->setup_solver();
    pso_solver_ptr->setup_operators(new CanonicalSpeedUpdate<float>(),
                                    new CanonicalPositionUpdate<float>());
    pso_solver_ptr->run();
    //pso_solver_ptr->print_solutions();
}
