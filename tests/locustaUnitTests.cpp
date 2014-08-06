#include "gtest/gtest.h"

#include <iostream>
#include <algorithm>
#include <time.h>

#include "benchmarks/benchmarks_gpu.h"

#include "prngenerator/prngenerator_gpu.h"
#include "population/population_set_gpu.h"

#include "evaluator/evaluator_gpu.h"
#include "solvers/ga/ga_solver_gpu.h"

using namespace locusta;

class TimedTest : public testing::Test {
protected:
    virtual void SetUp() {
        start_time = time(NULL);
    }

    virtual void TearDown() {
        const time_t end_time = time(NULL);
        const time_t elapsed_time = end_time - start_time;

        RecordProperty("Elapsed Time", elapsed_time);
        EXPECT_TRUE(elapsed_time >= 0);
    }

    time_t start_time;
};

class GPUPopulationSetupTest : public TimedTest {
    virtual void SetUp() {

        TimedTest::SetUp();
        __setup_cuda();

        upper_bounds = new float[DIM];
        lower_bounds = new float[DIM];

        std::fill(upper_bounds, upper_bounds+DIM, 1);
        std::fill(lower_bounds, lower_bounds+DIM, 1);

        population = new population_set_gpu<float>(ISLES,
                                                   AGENTS,
                                                   DIM,
                                                   upper_bounds,
                                                   lower_bounds);

        population->_initialize();

    }

    virtual void TearDown() {
        TimedTest::TearDown();

        delete population;

        delete [] upper_bounds;
        delete [] lower_bounds;
    }

public:

    population_set_gpu<float> * population;

    const size_t ISLES = 2;
    const size_t AGENTS = 10;
    const size_t DIM = 3;

    float * upper_bounds;
    float * lower_bounds;

};

TEST_F(GPUPopulationSetupTest, PopulationInit) {

    EXPECT_TRUE(population);
}
