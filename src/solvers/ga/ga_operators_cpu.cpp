namespace locusta {

    template<typename TFloat>
    inline void ga_operators_cpu<TFloat>::tournament_select
    (const bool F_SELF_SELECTION,
     const uint32_t SELECTION_SIZE,
     const TFloat SELECTION_P,
     const uint32_t NUM_ISLES,
     const uint32_t NUM_AGENTS,
     const TFloat * const fitness_array,
     uint32_t * selection_array,
     const TFloat * const prnumbers_array)
    {
        for(uint32_t i = 0; i < NUM_ISLES; ++i)
        {
            for(uint32_t j = 0; j < NUM_AGENTS; ++j)
            {
                const uint32_t idx = i * NUM_AGENTS + j;
                const uint32_t offset_size = NUM_ISLES * NUM_AGENTS;  // TODO: Check PRN cardinality and addressing!

                uint32_t candidates[SELECTION_SIZE];

                //Resevoir Sampling
                // Fill
                for (uint32_t k = 0; k < SELECTION_SIZE; ++k)
                {
                    if (F_SELF_SELECTION)
                    {
                        candidates[k] = k;
                    }
                    else
                    {
                        candidates[k] = k < j ? k : k + 1;
                    }
                }
                // Replace
                uint32_t random_idx;
                uint32_t iter_limit = F_SELF_SELECTION ? NUM_AGENTS : NUM_AGENTS - 1;
                uint32_t prn_idx = idx;

                for (uint32_t k = SELECTION_SIZE; k < iter_limit; ++k)
                {
                    random_idx = (NUM_AGENTS - 1) * prnumbers_array[prn_idx];
                    prn_idx += offset_size;
                    if (random_idx <= SELECTION_SIZE)
                    {
                        if (F_SELF_SELECTION)
                        {
                            candidates[random_idx] = k;
                        }
                        else
                        {
                            candidates[random_idx] = k < j ? k : k + 1;
                        }
                    }
                }

                // Tournament
                bool switch_flag;
                TFloat candidate_fitness_array;
                TFloat best_fitness_array = fitness_array[candidates[0]];

                for (uint32_t k = 1; k < SELECTION_SIZE; ++k)
                {
                    candidate_fitness_array = fitness_array[candidates[k]];
                    switch_flag = (candidate_fitness_array > best_fitness_array);

                    if ((SELECTION_P != 0.0f) &&
                        (SELECTION_P >= prnumbers_array[prn_idx]))
                    {
                        switch_flag = !switch_flag;
                    }

                    prn_idx += offset_size;

                    if (switch_flag)
                    {
                        best_fitness_array = candidate_fitness_array;
                        candidates[0] = candidates[k];
                    }
                }
                //std::cout << candidates[0] << " WON!" << std::endl;
                selection_array[idx] = candidates[0];
            }
        }
    }

    template<typename TFloat>
    inline void ga_operators_cpu<TFloat>::whole_crossover
    (const TFloat CROSSOVER_RATE,
     const TFloat MUTATION_RATE,
     const TFloat DIST_LIMIT,
     const TFloat DEVIATION,
     const uint32_t NUM_ISLES,
     const uint32_t NUM_AGENTS,
     const uint32_t NUM_DIMENSIONS,
     const TFloat * const UPPER_BOUNDS,
     const TFloat * const LOWER_BOUNDS,
     const TFloat * const VAR_RANGES,
     const TFloat * const parents_array,
     TFloat * offspring_array,
     const uint32_t * const coupling_array,
     const TFloat * const prnumbers_array,
     prngenerator<TFloat> * const local_generator)
    {
        TFloat INV_DIST_LIMIT = 1.0 / DIST_LIMIT;

        for (uint32_t i = 0; i < NUM_AGENTS; ++i)
        {
            TFloat * offspring_array_genome = &(offspring_array[i * NUM_DIMENSIONS]);
            const TFloat * genome_a = &(parents_array[i * NUM_DIMENSIONS]);
            const uint32_t & couple_idx = coupling_array[i];
            const TFloat * genome_b = &(parents_array[couple_idx * NUM_DIMENSIONS]);

            const uint32_t prn_mutation_offset = NUM_DIMENSIONS + 1;

            for (uint32_t j = 0; j < NUM_DIMENSIONS; ++j)
            {
                offspring_array_genome[j] = genome_a[j];
            }

            for (uint32_t j = 0; j < NUM_DIMENSIONS; ++j)
            {
                // Crossover
                if (prnumbers_array[i] < CROSSOVER_RATE)
                {
                    offspring_array_genome[j] *= 0.5;
                    offspring_array_genome[j] += 0.5 * genome_b[j];
                }

                // Mutation
                if (prnumbers_array[i*prn_mutation_offset + j] < MUTATION_RATE) // TODO: Check PRN Addressing
                {
                    const TFloat & range = VAR_RANGES[j];

                    TFloat x = 0.0;
                    for(uint32_t n = 0; n < DIST_LIMIT; ++n)
                    {
                        x += local_generator->_generate();
                    }
                    x *= INV_DIST_LIMIT;
                    x -= 0.5;
                    x *= DEVIATION * range;
                    x += offspring_array_genome[j];

                    // Bound Checking
                    const TFloat & u = UPPER_BOUNDS[j];
                    const TFloat & l = LOWER_BOUNDS[j];

                    if ( x > u )
                    {
                        offspring_array_genome[j] = u;
                    }
                    else if ( x < l )
                    {
                        offspring_array_genome[j] = l;
                    }
                    else
                    {
                        offspring_array_genome[j] = x;
                    }
                }
            }
        }
    }

    template<typename TFloat>
    inline void ga_operators_cpu<TFloat>::migration_ring
    (const uint32_t NUM_ISLES,
     const uint32_t NUM_AGENTS,
     const uint32_t NUM_DIMENSIONS,
     TFloat * data_array,
     TFloat * fitness_array,
     uint32_t * const migrating_idxs,
     TFloat * migration_buffer,
     const uint32_t MIGRATION_SIZE,
     const uint32_t MIGRATION_SELECTION_SIZE,
     prngenerator<TFloat> * const local_generator)
    {
        for(uint32_t iteration = 0; iteration < MIGRATION_SIZE; ++iteration)
        {
            // Migration Selection
            for(uint32_t i = 0; i < NUM_ISLES; ++i)
            {
                const uint32_t isle_offset = i * NUM_AGENTS;

                uint32_t migration_candidates_idxs[MIGRATION_SELECTION_SIZE];
                TFloat migration_candidates_fitness[MIGRATION_SELECTION_SIZE];

                // Resevoir Sampling
                // Fill
                for (uint32_t k = 0; k < MIGRATION_SELECTION_SIZE; ++k)
                {
                    migration_candidates_idxs[k] = k;
                }

                // Replace
                uint32_t random_idx;

                for (uint32_t k = MIGRATION_SELECTION_SIZE; k < NUM_AGENTS; ++k)
                {
                    random_idx = (NUM_AGENTS - 1) * local_generator->_generate();
                    if (random_idx <= MIGRATION_SELECTION_SIZE)
                    {
                        migration_candidates_idxs[random_idx] = k;
                    }
                }

                // Copy needed fitness values
                for (uint32_t k = 0; k < MIGRATION_SELECTION_SIZE; ++k)
                {
                    migration_candidates_fitness[k] = fitness_array[isle_offset + migration_candidates_idxs[k]];
                }

                // Tournament
                bool switch_flag;
                TFloat candidate_fitness;
                TFloat best_fitness = migration_candidates_fitness[0];

                for (uint32_t k = 1; k < MIGRATION_SELECTION_SIZE; ++k)
                {
                    candidate_fitness = migration_candidates_fitness[k];
                    switch_flag = (candidate_fitness > best_fitness);

                    if (switch_flag)
                    {
                        best_fitness = candidate_fitness;
                        migration_candidates_idxs[0] = migration_candidates_idxs[k];
                    }
                }
                migrating_idxs[iteration * NUM_ISLES + i] = migration_candidates_idxs[0];

                const TFloat * const genome =
                    data_array +
                    i * NUM_AGENTS * NUM_DIMENSIONS +
                    migration_candidates_idxs[0] * NUM_DIMENSIONS;

                TFloat * const buffer =
                    migration_buffer +
                    iteration * NUM_ISLES * (NUM_DIMENSIONS + 1);

                // Copy genomes to buffer
                for(uint32_t k = 0; k < NUM_DIMENSIONS; ++k)
                {
                    buffer[k] = genome[k];
                }

                buffer[NUM_DIMENSIONS] = best_fitness;
            }
        }

        // Migration
        for(uint32_t iteration = 0; iteration < MIGRATION_SIZE; ++iteration)
        {
            // Migration Selection
            for(uint32_t i = 0; i < NUM_ISLES; ++i)
            {
                const TFloat * const buffer =
                    migration_buffer +
                    iteration * NUM_ISLES * (NUM_DIMENSIONS + 1);

                TFloat * const genome =
                    data_array +
                    i * NUM_AGENTS * NUM_DIMENSIONS +
                    migrating_idxs[iteration * NUM_ISLES + i] * NUM_DIMENSIONS;

                for(uint32_t k = 0; k < NUM_DIMENSIONS; ++k)
                {
                    genome[k] = buffer[k];
                }
                fitness_array[i * NUM_AGENTS] = buffer[NUM_DIMENSIONS];
            }
        }
    }

} // namespace locusta
