namespace locusta {

    template<typename TFloat>
    prngenerator_cpu<TFloat>::prngenerator_cpu() : prngenerator<TFloat>::prngenerator(1)
    {
        _prng_engines = new mersenne_twister[_NUM_GENERATORS];
    }

    template<typename TFloat>
    prngenerator_cpu<TFloat>::prngenerator_cpu(uint32_t num_generators) : prngenerator<TFloat>::prngenerator(num_generators)
    {
        _prng_engines = new mersenne_twister[_NUM_GENERATORS];
    }

    template<typename TFloat>
    prngenerator_cpu<TFloat>::~prngenerator_cpu()
    {
        delete [] _prng_engines;
    }

    template<typename TFloat>
    void prngenerator_cpu<TFloat>::_initialize_engines(uint64_t seed)
    {
        if(seed == 0)
        {
            seed = std::chrono::system_clock::now().time_since_epoch().count();
        }

        std::minstd_rand0 seeder(seed);
        for(uint32_t i = 0; i < _NUM_GENERATORS; ++i)
        {
            TFloat local_seed = seeder();
            this->_prng_engines[i].seed(local_seed);
        }
    }

    template<typename TFloat>
    void prngenerator_cpu<TFloat>::_generate(const uint32_t n, TFloat * output)
    {
        const uint32_t chunk_size = n/_NUM_GENERATORS;

#pragma omp parallel num_threads(_NUM_GENERATORS) default(shared)
        {
            const int nthread = omp_get_thread_num();
            uni_real_dist real_dist;
            mersenne_twister &local_real_prng = this->_prng_engines[nthread];

#pragma omp for schedule(static, chunk_size)
            for(uint32_t i = 0; i < n; ++i)
            {
                output[i] = real_dist(local_real_prng);
            }
        }
    }

    template<typename TFloat>
    inline TFloat prngenerator_cpu<TFloat>::_generate()
    {
        const int nthread = omp_get_thread_num();
        uni_real_dist real_dist;
        mersenne_twister &local_real_prng = this->_prng_engines[nthread];

        return real_dist(local_real_prng);
    }

}
