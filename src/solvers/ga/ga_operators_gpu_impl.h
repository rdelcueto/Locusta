#include "cuda_common/cuda_helpers.h"

namespace locusta {

    /// Cuda Wrappers Forward Declarations
    template<typename TFloat>
        void
        tournament_select_dispatch
        (const bool F_SELF_SELECTION,
         const uint32_t SELECTION_SIZE,
         const TFloat SELECTION_P,
         const uint32_t NUM_ISLES,
         const uint32_t NUM_AGENTS,
         const TFloat * const fitness_array,
         uint32_t * selection_array,
         const TFloat * const prnumbers_array);

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
         prngenerator<TFloat> * const local_generator);

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
         prngenerator<TFloat> * const local_generator);

    template<typename TFloat>
        void ga_operators_gpu<TFloat>::tournament_select
        (const bool F_SELF_SELECTION,
         const uint32_t SELECTION_SIZE,
         const TFloat SELECTION_P,
         const uint32_t NUM_ISLES,
         const uint32_t NUM_AGENTS,
         const TFloat * const fitness_array,
         uint32_t * selection_array,
         const TFloat * const prnumbers_array)
    {
        tournament_select_dispatch<TFloat>
            (F_SELF_SELECTION,
             SELECTION_SIZE,
             SELECTION_P,
             NUM_ISLES,
             NUM_AGENTS,
             fitness_array,
             selection_array,
             prnumbers_array);
    }

    template<typename TFloat>
        void ga_operators_gpu<TFloat>::whole_crossover
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
        whole_crossover_dispatch
            (CROSSOVER_RATE,
             MUTATION_RATE,
             DIST_LIMIT,
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
             local_generator);
    }

    template<typename TFloat>
        void ga_operators_gpu<TFloat>::migration_ring
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
        migration_ring_dispatch<TFloat>
            (NUM_ISLES,
             NUM_AGENTS,
             NUM_DIMENSIONS,
             data_array,
             fitness_array,
             migrating_idxs,
             migration_buffer,
             MIGRATION_SIZE,
             MIGRATION_SELECTION_SIZE,
             local_generator);
    }

} /// namespace locusta
