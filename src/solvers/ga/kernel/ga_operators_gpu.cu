#include "../ga_operators_gpu.h"

namespace locusta {

    /// GPU Kernels Shared Memory Pointer.
    extern __shared__ int ga_shared_memory[];

    // ----------------------
    // GPU Kernel definitions
    // ----------------------
  
    template<typename TFloat>
    __global__
    void tournament_select_kernel
    (const bool F_SELF_SELECTION,
     const uint32_t SELECTION_SIZE,
     const TFloat SELECTION_P,
     const uint32_t NUM_ISLES,
     const uint32_t NUM_AGENTS,
     const TFloat * const fitness_array,
     uint32_t * selection_array,
     const TFloat * const prnumbers_array)
    {
        const uint32_t idx = blockIdx.x * NUM_AGENTS + threadIdx.x;
        const uint32_t offset_size = NUM_ISLES * NUM_AGENTS;
        uint32_t * candidates = selection_array + idx;

        //Resevoir Sampling
        // Fill
        for (uint32_t j = 0; j < SELECTION_SIZE; ++j)
        {
            if (F_SELF_SELECTION)
            {
                candidates[j*offset_size] = j;
            }
            else
            {
                candidates[j*offset_size] = j < threadIdx.x ? j : j + 1;
            }
        }
        // Replace
        uint32_t random_idx;
        uint32_t iter_limit = F_SELF_SELECTION ? NUM_AGENTS : NUM_AGENTS - 1;
        uint32_t prn_idx = blockIdx.x * NUM_AGENTS + threadIdx.x;

        // NOTE: prnnumbers required for section: NUM_ISLES * NUM_AGENTS * (NUM_AGENTS - SELECTION_SIZE).
        for (uint32_t j = SELECTION_SIZE; j < iter_limit; ++j)
        {
            random_idx = (NUM_AGENTS - 1) * prnumbers_array[prn_idx];        
            prn_idx += offset_size;
            if (random_idx <= SELECTION_SIZE)
            {
                if (F_SELF_SELECTION)
                {
                    candidates[random_idx*offset_size] = j;
                }
                else
                {
                    candidates[random_idx*offset_size] = j < threadIdx.x ? j : j + 1;
                }
            }
        }

        // Tournament        
        bool switch_flag;
        TFloat switch_p;
        TFloat candidate_fitness_array;
        TFloat best_fitness_array = fitness_array[candidates[0]];

        // NOTE: prnnumbers required for section: NUM_ISLES * NUM_AGENTS * (SELECTION_SIZE - 1) iff SELECTION_P.
        for (uint32_t j = 1; j < SELECTION_SIZE; ++j)
        {
            candidate_fitness_array = fitness_array[candidates[j*offset_size]];
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
                candidates[0] = candidates[j*offset_size];
            }
        }
        //std::cout << candidates[0] << " WON!" << std::endl;
        // TODO: Remove? Redundant with candidates[0]
        selection_array[idx] = candidates[0];
    }

    template<typename TFloat>
    __global__
    void whole_crossover_kernel
    (const TFloat CROSSOVER_RATE,
     const TFloat MUTATION_RATE,
     TFloat DIST_LIMIT,
     TFloat INV_DIST_LIMIT,
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
     curandState * const local_generator)
    {
        curandState local_state = local_generator[blockIdx.x * NUM_AGENTS + threadIdx.x];

        uint32_t prn_idx = blockIdx.x * NUM_AGENTS + threadIdx.x;
        uint32_t prn_offset = NUM_ISLES * NUM_AGENTS;

        const uint32_t genome_a_base_idx =
            blockIdx.x * NUM_DIMENSIONS * NUM_AGENTS + threadIdx.x;
    
        const uint32_t genome_b_base_idx =
            blockIdx.x * NUM_DIMENSIONS * NUM_AGENTS + coupling_array[threadIdx.x];

        const TFloat co_f = prnumbers_array[prn_idx];
        prn_idx += prn_offset;

        for (uint32_t i = 0; i < NUM_DIMENSIONS; ++i)
        {
            TFloat a = parents_array[genome_a_base_idx + i * NUM_AGENTS];
            TFloat b = parents_array[genome_b_base_idx + i * NUM_AGENTS];

            TFloat o;
        
            // Crossover
            if (co_f < CROSSOVER_RATE)
            {
                o = 0.5 * (a + b);
            }
            else
            {
                o = a;
            }

            // Mutation
            if (prnumbers_array[prn_idx] < MUTATION_RATE) // TODO: Check PRN Addressing
            {
                const TFloat & range = VAR_RANGES[i];
        
                TFloat x = 0.0;
                for(uint32_t n = 0; n < DIST_LIMIT; ++n)
                {
                    x += curand_uniform(&local_state);
                }
            
                x *= INV_DIST_LIMIT;
                x -= 0.5;
                x *= DEVIATION * range;
                x += o;

                // Bound Checking
                const TFloat & u = UPPER_BOUNDS[i];
                const TFloat & l = LOWER_BOUNDS[i];

                if ( x > u )
                {
                    o = u;
                }
                else if ( x < l )
                {
                    o = l;
                }
                else
                {
                    o = x;
                }
            }
            prn_idx += prn_offset;
            offspring_array[genome_a_base_idx + i * NUM_AGENTS] = o;
        }

        local_generator[blockIdx.x * NUM_AGENTS + threadIdx.x] = local_state;
    }

    template<typename TFloat>
    __global__
    void migration_selection_kernel
    (const uint32_t NUM_ISLES,
     const uint32_t NUM_AGENTS,
     const uint32_t NUM_DIMENSIONS,
     TFloat * data_array,
     TFloat * fitness_array,
     uint32_t * const migrating_idxs,
     const uint32_t MIGRATION_SIZE,
     const uint32_t MIGRATION_SELECTION_SIZE,
     curandState * const local_generator)
    {
        uint32_t * shr_reservoir = (uint32_t *) ga_shared_memory; // TODO: Check for overflow case.
        TFloat * shr_fitness = (TFloat *) &shr_reservoir[blockDim.x];

        curandState local_state = local_generator[threadIdx.x + blockIdx.x * MIGRATION_SELECTION_SIZE];
        uint32_t local_count = threadIdx.x + MIGRATION_SELECTION_SIZE;

        for(uint32_t iteration = 0; iteration < MIGRATION_SIZE; ++iteration)
        {
            // Reservoir Sampling
            // Fill
            shr_reservoir[threadIdx.x] = threadIdx.x;

            // Replace
            uint32_t local_rand;
            while(local_count < NUM_AGENTS)
            {
                local_rand = (NUM_AGENTS - 1) * curand_uniform(&local_state);
                if(local_rand <= MIGRATION_SELECTION_SIZE)
                {
                    shr_reservoir[local_rand] = local_count;
                }
                local_count += MIGRATION_SELECTION_SIZE;
                __syncthreads();
            }    

            // Get fitness value from Global Memory into Shared Memory
            shr_fitness[threadIdx.x] = fitness_array[shr_reservoir[threadIdx.x] + blockIdx.x * NUM_AGENTS];
            __syncthreads();
        
            // Tournament
            TFloat a, b;
            int reduce_idx = 1;
    
            const int reduce_limit = blockDim.x;
            while (reduce_idx < reduce_limit) reduce_idx <<= 1;

            while (reduce_idx != 0)
            {
                if (threadIdx.x < reduce_idx &&
                    (threadIdx.x + reduce_idx < reduce_limit))
                {
                    a = shr_fitness[threadIdx.x];
                    b = shr_fitness[threadIdx.x + reduce_idx];

                    if ( b > a )
                    {
                        shr_fitness[threadIdx.x] = b;
                        shr_reservoir[threadIdx.x] =
                            shr_reservoir[threadIdx.x + reduce_idx];
                    }
                }
                __syncthreads();
                reduce_idx >>= 1;
            }

            // Write Tournament Winner
            if ( threadIdx.x == 0 )
            {
                migrating_idxs[blockIdx.x + blockDim.x * iteration] = shr_reservoir[0];
                //printf("%d := %d, fitness = %f\n", blockIdx.x, shr_reservoir[0], fitness_array[shr_reservoir[0]]);
            }
        }

        local_generator[threadIdx.x + blockIdx.x * MIGRATION_SELECTION_SIZE] = local_state;
        return;
    }

    template<typename TFloat>
    __global__
    void migration_ring_kernel
    (const uint32_t NUM_ISLES,
     const uint32_t NUM_AGENTS,
     const uint32_t NUM_DIMENSIONS,
     TFloat * data_array,
     TFloat * fitness_array,
     TFloat * migration_buffer,
     uint32_t * const migrating_idxs,
     const uint32_t MIGRATION_SIZE)
    {
        const uint32_t isle = blockIdx.x;
        const uint32_t target_isle = blockIdx.x + 1 < NUM_ISLES ? blockIdx.x + 1 : 0;

        TFloat * migration_fitness = &migration_buffer[NUM_ISLES * NUM_DIMENSIONS * MIGRATION_SIZE];

        for(uint32_t i = 0; i < MIGRATION_SIZE; ++i)
        {
            migration_fitness[isle + i * NUM_ISLES] =
                fitness_array[migrating_idxs[isle] + blockIdx.x * NUM_AGENTS];
        
            const uint32_t gene_base_idx =
                blockIdx.x * NUM_DIMENSIONS * NUM_AGENTS + (migrating_idxs[isle]);
        
            for(uint32_t d = 0; d < NUM_DIMENSIONS; ++d)
            {
                migration_buffer[isle + d * NUM_ISLES + i * NUM_DIMENSIONS * NUM_ISLES] =
                    data_array[gene_base_idx + d * NUM_AGENTS];
            }
            __syncthreads();
        
            fitness_array[migrating_idxs[isle] + blockIdx.x * NUM_AGENTS] =
                migration_fitness[target_isle + i * NUM_ISLES];
        
            for(uint32_t d = 0; d < NUM_DIMENSIONS; ++d)
            {
                data_array[gene_base_idx + d * NUM_AGENTS] = migration_buffer[target_isle + d * NUM_ISLES];
            }
        }
    }
  
    // -------------
    // CUDA Wrappers
    // -------------
  
    template<typename TFloat>
    void tournament_select_dispatch
    (const bool F_SELF_SELECTION,
     const uint32_t SELECTION_SIZE,
     const TFloat SELECTION_P,
     const uint32_t NUM_ISLES,
     const uint32_t NUM_AGENTS,
     const TFloat * const fitness_array,
     uint32_t * selection_array,
     const TFloat * const prnumbers_array)
    {
        tournament_select_kernel
            <<<NUM_ISLES, NUM_AGENTS>>>
            (F_SELF_SELECTION,
             SELECTION_SIZE,
             SELECTION_P,
             NUM_ISLES,
             NUM_AGENTS,
             fitness_array,
             selection_array,
             prnumbers_array);
        CudaCheckError();
    }

    template<typename TFloat>
    void whole_crossover_dispatch
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

        prngenerator_gpu<TFloat> * const dev_local_generator = static_cast<prngenerator_gpu<TFloat>*>(local_generator);
        const TFloat INV_DIST_LIMIT = 1.0 / DIST_LIMIT;
    
        whole_crossover_kernel
            <<<NUM_ISLES, NUM_AGENTS>>>
            (CROSSOVER_RATE,
             MUTATION_RATE,
             DIST_LIMIT,
             INV_DIST_LIMIT,
             DEVIATION,
             NUM_ISLES,
             NUM_AGENTS,
             NUM_DIMENSIONS,
             UPPER_BOUNDS,
             LOWER_BOUNDS,
             VAR_RANGES,
             parents_array,
             offspring_array,
             coupling_array,
             prnumbers_array,
             dev_local_generator->get_device_generator_states());
    
        CudaCheckError();
    }

    template<typename TFloat>
    void migration_ring_dispatch
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
        prngenerator_gpu<TFloat> * const dev_local_generator = static_cast<prngenerator_gpu<TFloat>*>(local_generator);
    
        migration_selection_kernel
            <<<NUM_ISLES, MIGRATION_SELECTION_SIZE, MIGRATION_SELECTION_SIZE * (sizeof(uint32_t) + sizeof(TFloat))>>>
            (NUM_ISLES,
             NUM_AGENTS,
             NUM_DIMENSIONS,
             data_array,
             fitness_array,
             migrating_idxs,
             MIGRATION_SIZE,
             MIGRATION_SELECTION_SIZE,
             dev_local_generator->get_device_generator_states());
        CudaCheckError();

        migration_ring_kernel
            <<<NUM_ISLES, NUM_AGENTS>>>
            (NUM_ISLES,
             NUM_AGENTS,
             NUM_DIMENSIONS,
             data_array,
             fitness_array,
             migration_buffer,
             migrating_idxs,
             MIGRATION_SIZE);
        CudaCheckError();
    }

    // -------------------------------
    // Template Specialization (float)
    // -------------------------------
  
    template
    void tournament_select_dispatch<float>
    (const bool F_SELF_SELECTION,
     const uint32_t SELECTION_SIZE,
     const float SELECTION_P,
     const uint32_t NUM_ISLES,
     const uint32_t NUM_AGENTS,
     const float * const fitness_array,
     uint32_t * selection_array,
     const float * const prnumbers_array);

    template
    void whole_crossover_dispatch<float>
    (const float CROSSOVER_RATE,
     const float MUTATION_RATE,
     const float DIST_LIMIT,
     const float DEVIATION,
     const uint32_t NUM_ISLES,
     const uint32_t NUM_AGENTS,
     const uint32_t NUM_DIMENSIONS,
     const float * const UPPER_BOUNDS,
     const float * const LOWER_BOUNDS,
     const float * const VAR_RANGES,
     const float * const parents_array,
     float * offspring_array,
     const uint32_t * const coupling_array,
     const float * const prnumbers_array,
     prngenerator<float> * const local_generator);

    template
    void migration_ring_dispatch<float>
    (const uint32_t NUM_ISLES,
     const uint32_t NUM_AGENTS,
     const uint32_t NUM_DIMENSIONS,
     float * data_array,
     float * fitness_array,
     uint32_t * const migrating_idxs,
     float * migration_buffer,
     const uint32_t MIGRATION_SIZE,
     const uint32_t MIGRATION_SELECTION_SIZE,
     prngenerator<float> * const local_generator);

    // --------------------------------
    // Template Specialization (double)
    // --------------------------------
  
    template
    void tournament_select_dispatch<double>
    (const bool F_SELF_SELECTION,
     const uint32_t SELECTION_SIZE,
     const double SELECTION_P,
     const uint32_t NUM_ISLES,
     const uint32_t NUM_AGENTS,
     const double * const fitness_array,
     uint32_t * selection_array,
     const double * const prnumbers_array);

    template
    void whole_crossover_dispatch<double>
    (const double CROSSOVER_RATE,
     const double MUTATION_RATE,
     const double DIST_LIMIT,
     const double DEVIATION,
     const uint32_t NUM_ISLES,
     const uint32_t NUM_AGENTS,
     const uint32_t NUM_DIMENSIONS,
     const double * const UPPER_BOUNDS,
     const double * const LOWER_BOUNDS,
     const double * const VAR_RANGES,
     const double * const parents_array,
     double * offspring_array,
     const uint32_t * const coupling_array,
     const double * const prnumbers_array,
     prngenerator<double> * const local_generator);

    template
    void migration_ring_dispatch<double>
    (const uint32_t NUM_ISLES,
     const uint32_t NUM_AGENTS,
     const uint32_t NUM_DIMENSIONS,
     double * data_array,
     double * fitness_array,
     uint32_t * const migrating_idxs,
     double * migration_buffer,
     const uint32_t MIGRATION_SIZE,
     const uint32_t MIGRATION_SELECTION_SIZE,
     prngenerator<double> * const local_generator);
  
} // namespace locusta
