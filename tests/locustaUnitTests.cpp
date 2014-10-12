#include "gtest/gtest.h"

#include <iostream>
#include <algorithm>
#include <time.h>

#include "benchmarks/benchmarks_cpu.hpp"
#include "benchmarks/benchmarks_gpu.hpp"

#include "prngenerator/prngenerator_cpu.hpp"
#include "prngenerator/prngenerator_gpu.hpp"

#include "population/population_set.hpp"
#include "population/population_set_gpu.hpp"

#include "evaluator/evaluator_cpu.hpp"
#include "evaluator/evaluator_gpu.hpp"

#include "solvers/pso/pso_solver.hpp"
#include "solvers/pso/pso_operators/pso_std_operators.hpp"

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

            prngenerator_cpu_ptr = new prngenerator_cpu<float>(ISLES * AGENTS);
            prngenerator_cpu_ptr->_initialize_engines(SEED);

            prngenerator_gpu_ptr = new prngenerator_gpu<float>(ISLES * AGENTS);
            prngenerator_gpu_ptr->_initialize_engines(SEED);

            evaluator_cpu_ptr = new evaluator_cpu<float>(true,
                                                         evaluator_cpu<float>::IGNORE_BOUNDS,
                                                         0,
                                                         benchmark_cpu_func_1<float>);

            evaluator_gpu_ptr = new evaluator_gpu<float>(true,
                                                         evaluator_gpu<float>::IGNORE_BOUNDS,
                                                         0,
                                                         benchmark_gpu_func_1<float>);

            population_cpu_ptr = new population_set<float>(ISLES, AGENTS, DIMENSIONS);
            population_gpu_ptr = new population_set_gpu<float>(ISLES, AGENTS, DIMENSIONS);

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
            delete population_gpu_ptr;

            delete evaluator_cpu_ptr;
            delete evaluator_gpu_ptr;

            delete prngenerator_cpu_ptr;
            delete prngenerator_gpu_ptr;

            RecordProperty("Elapsed Time", elapsed_time);
        }

    time_t start_time;

    // Pseudo Random Number Generators
    prngenerator_cpu<float> * prngenerator_cpu_ptr;
    prngenerator_gpu<float> * prngenerator_gpu_ptr;

    // Evaluator
    evaluator_cpu<float> * evaluator_cpu_ptr;
    evaluator_gpu<float> * evaluator_gpu_ptr;

    // Population
    const uint64_t SEED = 1;
    const size_t GENERATIONS = 1e2;
    const size_t ISLES = 1;
    const size_t AGENTS = 128;
    const size_t DIMENSIONS = 128;

    population_set<float> * population_cpu_ptr;
    population_set_gpu<float> * population_gpu_ptr;

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
