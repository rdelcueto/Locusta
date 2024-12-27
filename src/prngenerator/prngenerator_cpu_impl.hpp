#include "prngenerator_cpu.hpp"

namespace locusta {

/**
 * @brief Construct a new prngenerator_cpu object with a single engine.
 */
template<typename TFloat>
prngenerator_cpu<TFloat>::prngenerator_cpu()
  : prngenerator<TFloat>::prngenerator(1)
{
  _prng_engines = new mersenne_twister[_NUM_ENGINES];
  _prng_distributions = new uni_real_distribution[_NUM_ENGINES];
}

/**
 * @brief Construct a new prngenerator_cpu object.
 *
 * @param num_engines Number of PRNG engines to use.
 */
template<typename TFloat>
prngenerator_cpu<TFloat>::prngenerator_cpu(uint32_t num_engines)
  : prngenerator<TFloat>::prngenerator(num_engines)
{
  _prng_engines = new mersenne_twister[_NUM_ENGINES];
  _prng_distributions = new uni_real_distribution[_NUM_ENGINES];
}

/**
 * @brief Destroy the prngenerator_cpu object.
 */
template<typename TFloat>
prngenerator_cpu<TFloat>::~prngenerator_cpu()
{
  delete[] _prng_distributions;
  delete[] _prng_engines;
}

/**
 * @brief Initialize the state of the PRNG engines.
 *
 * @param seed Seed value for the PRNG engines.
 */
template<typename TFloat>
void
prngenerator_cpu<TFloat>::_initialize_engines(uint64_t seed)
{
  if (seed == 0) {
    seed = std::chrono::system_clock::now().time_since_epoch().count();
  }

  for (uint32_t i = 0; i < _NUM_ENGINES; ++i) {
    std::minstd_rand0 seeder(seed + i);
    TFloat local_seed = seeder();
    this->_prng_engines[i].seed(local_seed);
    this->_prng_distributions[i];
  }
}

/**
 * @brief Generate pseudo-random numbers.
 *
 * This method generates \p n pseudo-random numbers and stores them in the \p
 * output array.
 *
 * @param n Number of pseudo-random numbers to generate.
 * @param output Output array to store the generated pseudo-random numbers.
 */
template<typename TFloat>
void
prngenerator_cpu<TFloat>::_generate(const uint32_t n, TFloat* output)
{

#pragma omp parallel firstprivate(n) shared(output)
  {
    const int nthread = omp_get_thread_num();
    uni_real_distribution& local_real_distribution =
      this->_prng_distributions[nthread];
    mersenne_twister& local_prng_engine = this->_prng_engines[nthread];

#pragma omp for
    for (uint32_t i = 0; i < n; ++i) {
      output[i] = local_real_distribution(local_prng_engine);
    }
  }
}
}
