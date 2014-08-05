#ifndef LOCUSTA_GA_OPERATORS_H
#define LOCUSTA_GA_OPERATORS_H

#include <functional>
#include "../../prngenerator/prngenerator.h"

namespace locusta {

    //Interface for Genetic Algorithm Operators
    template<typename TFloat>
        class ga_operators {
    public:

        typedef void (*select_func)(const bool F_SELF_SELECTION,
                                    const uint32_t SELECTION_SIZE,
                                    const TFloat SELECTION_P,
                                    const uint32_t NUM_ISLES,
                                    const uint32_t NUM_AGENTS,
                                    const TFloat * const fitness_array,
                                    uint32_t * selection_array,
                                    const TFloat * const prnumbers_array);

        typedef void (*breed_func)(const TFloat CROSSOVER_RATE,
                                   const TFloat MUTATION_RATE,
                                   const TFloat DIST_LIMIT,
                                   const TFloat DEVATION,
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

        typedef void (*migrate_func)(const uint32_t NUM_ISLES,
                                     const uint32_t NUM_AGENTS,
                                     const uint32_t NUM_DIMENSIONS,
                                     TFloat * data_array,
                                     TFloat * fitness_array,
                                     uint32_t * const migrating_idxs,
                                     TFloat * migration_buffer,
                                     const uint32_t MIGRATION_SIZE,
                                     const uint32_t MIGRATION_SELECTION_SIZE,
                                     prngenerator<TFloat> * const local_generator);

    };

} // namespace locusta
#endif
