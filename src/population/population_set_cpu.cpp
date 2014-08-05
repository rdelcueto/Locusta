namespace locusta {

    template <typename TFloat>
    population_set_cpu<TFloat>::population_set_cpu (const uint32_t NUM_ISLES,
                                                                      const uint32_t NUM_AGENTS,
                                                                      const uint32_t NUM_DIMENSIONS,
                                                                      TFloat * upper_bound,
                                                                      TFloat * lower_bound)
        : population_set<TFloat>(NUM_ISLES,
                                          NUM_AGENTS,
                                          NUM_DIMENSIONS,
                                          upper_bound,
                                          lower_bound)
    {
        _var_ranges = new TFloat[NUM_DIMENSIONS];

        _data_array = new TFloat[NUM_DIMENSIONS * NUM_AGENTS * NUM_ISLES];
        _transformed_data_array = new TFloat[NUM_DIMENSIONS * NUM_AGENTS * NUM_ISLES];
        _fitness_array = new TFloat[NUM_AGENTS * NUM_ISLES];

        _highest_idx = new uint32_t[NUM_ISLES];
        _lowest_idx = new uint32_t[NUM_ISLES];

        _highest_fitness = new TFloat[NUM_ISLES];
        _lowest_fitness = new TFloat[NUM_ISLES];
    }

    template <typename TFloat>
    population_set_cpu<TFloat>::~population_set_cpu()
    {
        delete [] _var_ranges;

        delete [] _data_array;
        delete [] _transformed_data_array;

        delete [] _fitness_array;

        delete [] _highest_idx;
        delete [] _lowest_idx;

        delete [] _highest_fitness;
        delete [] _lowest_fitness;
    }


    template <typename TFloat>
    void population_set_cpu<TFloat>::_initialize()
    {
        // Value Initialization
        for(uint32_t i = 0; i < _NUM_DIMENSIONS; ++i)
        {
            _var_ranges[i] = _UPPER_BOUNDS[i] - _LOWER_BOUNDS[i];
        }

        for(uint32_t i = 0; i < _NUM_ISLES; ++i)
        {
            _highest_idx[i] = 0;
            _lowest_idx[i] = 0;
            _highest_fitness[i] = -std::numeric_limits<TFloat>::infinity();
            _lowest_fitness[i] = std::numeric_limits<TFloat>::infinity();
        }

#pragma omp parallel for schedule(static)
        for(uint32_t i = 0; i < _NUM_ISLES; ++i)
        {
            for(uint32_t j = 0; j < _NUM_AGENTS; ++j)
            {
                for(uint32_t k = 0; k < _NUM_DIMENSIONS; ++k)
                {
                    // Initialize Data Array
                    const uint32_t data_idx =
                        i * _NUM_AGENTS * _NUM_DIMENSIONS +
                        j * _NUM_DIMENSIONS +
                        k;

                    _data_array[data_idx] =
                        _LOWER_BOUNDS[k] + (_var_ranges[k] * _transformed_data_array[data_idx]);
                    // Initialize Transformed Data Array with infinity values.
                    _transformed_data_array[data_idx] = -std::numeric_limits<TFloat>::infinity();
                }
                // Initialize Fitness Values
                const uint32_t fitness_idx = i * _NUM_AGENTS + j;
                _fitness_array[fitness_idx] = -std::numeric_limits<TFloat>::infinity();
            }
        }
    }

    template <typename TFloat>
    void population_set_cpu<TFloat>::_print_data()
    {
        // Update fitness values
        for(uint32_t isle_count = 0; isle_count < _NUM_ISLES; ++isle_count)
        {
            std::cout << "Isle: " << isle_count << std::endl;
            for(uint32_t agent_count = 0; agent_count < _NUM_AGENTS; ++agent_count)
            {
                TFloat * agent_fitness = _fitness_array +
                    isle_count * _NUM_AGENTS +
                    agent_count;

                TFloat * agent_data = _data_array +
                    isle_count * _NUM_AGENTS * _NUM_DIMENSIONS +
                    agent_count * _NUM_DIMENSIONS;

                std::cout << "\tAgent: " << agent_count <<
                    ", F: "<< agent_fitness[0] <<
                    "\n\t";
                for (uint32_t k = 0; k < _NUM_DIMENSIONS; ++k)
                {
                    std::cout << agent_data[k] << ", ";
                }
                std::cout << std::endl;
            }
        }
    }

    template <typename TFloat>
    void population_set_cpu<TFloat>::_swap_data_sets()
    {
        TFloat * const _tmp_data_pointer = _data_array;
        _data_array = _transformed_data_array;
        _transformed_data_array = _tmp_data_pointer;
    }

    template <typename TFloat>
    void population_set_cpu<TFloat>::_update_records()
    {
        const uint32_t ISLES = _NUM_ISLES;
        const uint32_t AGENTS = _NUM_AGENTS;
        //const uint32_t DIMENSIONS = _NUM_DIMENSIONS;

        // Update per Isle Records
#pragma omp parallel for schedule(static)
        for(uint32_t i = 0; i < ISLES; ++i)
        {
            const uint32_t local_offset = i * AGENTS;

            TFloat local_highest_value = _fitness_array[local_offset];
            uint32_t local_highest_idx = local_offset;

            TFloat local_lowest_value = _fitness_array[local_offset];
            uint32_t local_lowest_idx = local_offset;

            for(uint32_t j = 1; j < AGENTS; ++j)
            {
                const TFloat candidate = _fitness_array[local_offset + j];
                if (candidate > local_highest_value)
                {
                    local_highest_idx = local_offset + j;
                    local_highest_value = candidate;
                }

                if (candidate < local_lowest_value)
                {
                    local_lowest_idx = local_offset + j;
                    local_lowest_value = candidate;
                }
            }

            _highest_idx[i] = local_highest_idx;
            _highest_fitness[i] = local_highest_value;

            _lowest_idx[i] = local_lowest_idx;
            _lowest_fitness[i] = local_lowest_value;
        }

        // Update per Global Records
        TFloat elite_highest_value = _highest_fitness[0];
        uint32_t elite_highest_idx = _highest_idx[0];
        TFloat elite_lowest_value = _lowest_fitness[0];
        uint32_t elite_lowest_idx = _lowest_idx[0];

        for(uint32_t i = 1; i < _NUM_ISLES; ++i)
        {
            const TFloat hcandidate = _highest_fitness[i];
            if (hcandidate > elite_highest_value)
            {
                elite_highest_value = hcandidate;
                elite_highest_idx = i;
            }

            const TFloat lcandidate = _lowest_fitness[i];
            if (lcandidate > elite_lowest_value)
            {
                elite_lowest_value = lcandidate;
                elite_lowest_idx = i;
            }
        }

        _global_highest_fitness = elite_highest_value;
        _global_highest_idx = _highest_idx[elite_highest_idx];

        _global_lowest_fitness = elite_lowest_value;
        _global_lowest_idx = _lowest_idx[elite_lowest_idx];
    }

}
