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

#include "solvers/ga/ga_solver_cpu.hpp"
#include "solvers/ga/ga_solver_cuda.hpp"

#include "solvers/ga/ga_operators/ga_std_operators_cpu_impl.hpp"
#include "solvers/ga/ga_operators/ga_std_operators_cuda_impl.hpp"

#include "cuda_runtime.h"

using namespace locusta;

class LocustaTestEnvironment : public testing::Environment {
public:
    virtual ~LocustaTestEnvironment() {}
    // Override this to define how to set up the environment.
    virtual void SetUp() {
        upper_bounds_ptr = new float[DIMENSIONS];
        lower_bounds_ptr = new float[DIMENSIONS];

        // Bounds definition
        std::fill(upper_bounds_ptr, upper_bounds_ptr + DIMENSIONS, 100.0f);
        std::fill(lower_bounds_ptr, lower_bounds_ptr + DIMENSIONS, -100.0f);

        // Init timer
        start_time = time(NULL);
    }
    // Override this to define how to tear down the environment.
    virtual void TearDown() {
        const time_t end_time = time(NULL);
        const time_t elapsed_time = end_time - start_time;

        std::cout << "Elapsed Time" << elapsed_time << std::endl;
        //RecordProperty("Elapsed Time", elapsed_time);
    }

public:

    time_t start_time;

    const uint32_t BENCHMARK_FUNC_ID = 1;
    const uint64_t SEED = 1;
    const uint32_t GENERATIONS = 2e2;
    const uint32_t ISLES = 2;
    const uint32_t AGENTS = 8;
    const uint32_t DIMENSIONS = 8;

    float * upper_bounds_ptr;
    float * lower_bounds_ptr;

};

LocustaTestEnvironment* const locusta_glb_env = new LocustaTestEnvironment;
::testing::Environment* const locusta_env = ::testing::AddGlobalTestEnvironment(locusta_glb_env);

class CPULocustaTest : public testing::Test {
protected:
    virtual void SetUp() {

        evaluation_functor_cpu_ptr = new BenchmarkFunctor<float>(BENCHMARK_FUNC_ID, DIMENSIONS);
        evaluator_cpu_ptr = new evaluator_cpu<float>(evaluation_functor_cpu_ptr,
                                                     true,
                                                     BoundMapKind::CropBounds,
                                                     DIMENSIONS);

        prngenerator_cpu_ptr = new prngenerator_cpu<float>(omp_get_max_threads());
        prngenerator_cpu_ptr = new prngenerator_cpu<float>(1);
        prngenerator_cpu_ptr->_initialize_engines(SEED);

        population_cpu_ptr = new population_set_cpu<float>(ISLES, AGENTS, DIMENSIONS);
    }

    virtual void TearDown() {
        delete population_cpu_ptr;
        population_cpu_ptr = NULL;
        delete evaluator_cpu_ptr;
        evaluator_cpu_ptr = NULL;
        delete evaluation_functor_cpu_ptr;
        evaluation_functor_cpu_ptr = NULL;
        delete prngenerator_cpu_ptr;
        prngenerator_cpu_ptr = NULL;
    }

public:

    const uint32_t BENCHMARK_FUNC_ID = locusta_glb_env->BENCHMARK_FUNC_ID;
    const uint32_t SEED = locusta_glb_env->SEED;
    const uint32_t GENERATIONS = locusta_glb_env->GENERATIONS;
    const uint32_t ISLES = locusta_glb_env->ISLES;
    const uint32_t AGENTS = locusta_glb_env->AGENTS;
    const uint32_t DIMENSIONS = locusta_glb_env->DIMENSIONS;

    float * upper_bounds_ptr = locusta_glb_env->upper_bounds_ptr;
    float * lower_bounds_ptr = locusta_glb_env->lower_bounds_ptr;

    // Pseudo Random Number Generators
    prngenerator_cpu<float> * prngenerator_cpu_ptr;
    // Evaluator
    evaluator_cpu<float> * evaluator_cpu_ptr;
    EvaluationFunctor<float> * evaluation_functor_cpu_ptr;
    // Population
    population_set_cpu<float> * population_cpu_ptr;
};

class GPULocustaTest : public testing::Test {
protected:
    virtual void SetUp() {
        __setup_cuda();

        evaluation_functor_cuda_ptr = new BenchmarkCudaFunctor<float>(BENCHMARK_FUNC_ID, DIMENSIONS);
        evaluator_cuda_ptr = new evaluator_cuda<float>(evaluation_functor_cuda_ptr,
                                                       true,
                                                       BoundMapKind::CropBounds,
                                                       DIMENSIONS);

        prngenerator_cuda_ptr = new prngenerator_cuda<float>(ISLES * AGENTS);
        prngenerator_cuda_ptr->_initialize_engines(SEED);

        population_cuda_ptr = new population_set_cuda<float>(ISLES, AGENTS, DIMENSIONS);
    }

    virtual void TearDown() {
        delete population_cuda_ptr;
        population_cuda_ptr = NULL;
        delete evaluator_cuda_ptr;
        evaluator_cuda_ptr = NULL;
        delete evaluation_functor_cuda_ptr;
        evaluation_functor_cuda_ptr = NULL;
        delete prngenerator_cuda_ptr;
        prngenerator_cuda_ptr = NULL;
    }

public:

    const uint32_t BENCHMARK_FUNC_ID = locusta_glb_env->BENCHMARK_FUNC_ID;
    const uint32_t SEED = locusta_glb_env->SEED;
    const uint32_t GENERATIONS = locusta_glb_env->GENERATIONS;
    const uint32_t ISLES = locusta_glb_env->ISLES;
    const uint32_t AGENTS = locusta_glb_env->AGENTS;
    const uint32_t DIMENSIONS = locusta_glb_env->DIMENSIONS;

    float * upper_bounds_ptr = locusta_glb_env->upper_bounds_ptr;
    float * lower_bounds_ptr = locusta_glb_env->lower_bounds_ptr;

    // Pseudo Random Number Generators
    prngenerator_cuda<float> * prngenerator_cuda_ptr;
    // Evaluator
    EvaluationCudaFunctor<float> * evaluation_functor_cuda_ptr;
    evaluator_cuda<float> * evaluator_cuda_ptr;
    // Population
    population_set_cuda<float> * population_cuda_ptr;
};

class CPUParticleSwarmTest : public CPULocustaTest {
protected:
    virtual void SetUp() {

        CPULocustaTest::SetUp();
        pso_solver_cpu_ptr = new pso_solver_cpu<float>(population_cpu_ptr,
                                                       evaluator_cpu_ptr,
                                                       prngenerator_cpu_ptr,
                                                       GENERATIONS,
                                                       upper_bounds_ptr,
                                                       lower_bounds_ptr);
    }

    virtual void TearDown() {
        delete pso_solver_cpu_ptr;
        pso_solver_cpu_ptr = NULL;
        CPULocustaTest::TearDown();
    }

public:
    pso_solver_cpu<float> * pso_solver_cpu_ptr;

};

class GPUParticleSwarmTest : public GPULocustaTest {
protected:
    virtual void SetUp() {
        GPULocustaTest::SetUp();
        pso_solver_cuda_ptr = new pso_solver_cuda<float>(population_cuda_ptr,
                                                         evaluator_cuda_ptr,
                                                         prngenerator_cuda_ptr,
                                                         GENERATIONS,
                                                         upper_bounds_ptr,
                                                         lower_bounds_ptr);
    }

    virtual void TearDown() {
        delete pso_solver_cuda_ptr;
        pso_solver_cuda_ptr = NULL;
        GPULocustaTest::TearDown();
    }

public:
    pso_solver_cuda<float> * pso_solver_cuda_ptr;

};

class CPUGeneticAlgorithmTest : public CPULocustaTest {
protected:
    virtual void SetUp() {
        CPULocustaTest::SetUp();
        ga_solver_cpu_ptr = new ga_solver_cpu<float>(population_cpu_ptr,
                                                     evaluator_cpu_ptr,
                                                     prngenerator_cpu_ptr,
                                                     GENERATIONS,
                                                     upper_bounds_ptr,
                                                     lower_bounds_ptr);
    }

    virtual void TearDown() {
        delete ga_solver_cpu_ptr;
        ga_solver_cpu_ptr = NULL;
        CPULocustaTest::TearDown();
    }

public:
    ga_solver_cpu<float> * ga_solver_cpu_ptr;

};

class GPUGeneticAlgorithmTest : public GPULocustaTest {
protected:
    virtual void SetUp() {
        GPULocustaTest::SetUp();
        ga_solver_cuda_ptr = new ga_solver_cuda<float>(population_cuda_ptr,
                                                       evaluator_cuda_ptr,
                                                       prngenerator_cuda_ptr,
                                                       GENERATIONS,
                                                       upper_bounds_ptr,
                                                       lower_bounds_ptr);
    }

    virtual void TearDown() {
        delete ga_solver_cuda_ptr;
        ga_solver_cuda_ptr = NULL;
        GPULocustaTest::TearDown();
    }

public:
    ga_solver_cuda<float> * ga_solver_cuda_ptr;

// Benchmark Setup
};

TEST_F(CPUParticleSwarmTest, BenchmarkCpu) {
    pso_solver_cpu_ptr->setup_operators(new CanonicalParticleRecordUpdate<float>(),
                                        new CanonicalSpeedUpdate<float>(),
                                        new CanonicalPositionUpdate<float>());
    pso_solver_cpu_ptr->setup_solver();
    pso_solver_cpu_ptr->run();
    //pso_solver_cpu_ptr->print_solutions();
}

TEST_F(CPUGeneticAlgorithmTest, BenchmarkCpu) {
    ga_solver_cpu_ptr->setup_operators(new WholeCrossover<float>(),
                                       new TournamentSelection<float>());
    ga_solver_cpu_ptr->setup_solver();
    ga_solver_cpu_ptr->run();
    //ga_solver_cpu_ptr->print_solutions();
}

TEST_F(GPUParticleSwarmTest, BenchmarkCuda) {
    pso_solver_cuda_ptr->setup_operators(new CanonicalParticleRecordUpdateCuda<float>(),
                                         new CanonicalSpeedUpdateCuda<float>(),
                                         new CanonicalPositionUpdateCuda<float>());
    pso_solver_cuda_ptr->setup_solver();
    pso_solver_cuda_ptr->run();
    // pso_solver_cuda_ptr->print_population();
}

TEST_F(GPUGeneticAlgorithmTest, BenchmarkCuda) {
    ga_solver_cuda_ptr->setup_operators(new WholeCrossoverCuda<float>(),
                                        new TournamentSelectionCuda<float>());
    ga_solver_cuda_ptr->setup_solver();
    ga_solver_cuda_ptr->run();
    // ga_solver_cuda_ptr->print_population();
}

// int main(int argc, char **argv) {
//     ::testing::InitGoogleTest(&argc, argv);
//     ::testing::AddGlobalTestEnvironment(new Environment());
//     return RUN_ALL_TESTS();
// }
