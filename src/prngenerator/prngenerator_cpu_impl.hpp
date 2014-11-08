#include "prngenerator_cpu.hpp"

namespace locusta {

    template<typename TFloat>
    prngenerator_cpu<TFloat>::prngenerator_cpu() : prngenerator<TFloat>::prngenerator(1) {
        _prng_engines = new mersenne_twister[_NUM_ENGINES];
    }

    template<typename TFloat>
    prngenerator_cpu<TFloat>::prngenerator_cpu(uint32_t num_engines) : prngenerator<TFloat>::prngenerator(num_engines) {
        _prng_engines = new mersenne_twister[_NUM_ENGINES];
    }

    template<typename TFloat>
    prngenerator_cpu<TFloat>::~prngenerator_cpu() {
    }

    template<typename TFloat>
    void prngenerator_cpu<TFloat>::_initialize_engines(uint64_t seed) {
        if(seed == 0) {
            seed = std::chrono::system_clock::now().time_since_epoch().count();
        }

        for(uint32_t i = 0; i < _NUM_ENGINES; ++i) {
            std::minstd_rand0 seeder(seed+i);
            TFloat local_seed = seeder();
            this->_prng_engines[i].seed(local_seed);
        }
    }

    template<typename TFloat>
    void prngenerator_cpu<TFloat>::_generate(const uint32_t n, TFloat * output) {
//#pragma omp parallel for
        for(uint32_t i = 0; i < n; ++i) {
            output[i] = _generate();
        }
    }

    template<typename TFloat>
    inline TFloat prngenerator_cpu<TFloat>::_generate() {
        const int nthread = omp_get_thread_num();
        uni_real_dist real_dist;
        mersenne_twister &local_real_prng = this->_prng_engines[nthread];

        return real_dist(local_real_prng);
    }

}
