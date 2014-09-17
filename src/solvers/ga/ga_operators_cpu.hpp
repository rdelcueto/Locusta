#ifndef LOCUSTA_GA_OPERATORS_CPU_H
#define LOCUSTA_GA_OPERATORS_CPU_H

#include "ga_operators.hpp"

namespace locusta {

  template<typename TFloat>
  class ga_operators_cpu : public ga_operators<TFloat> {
  public:

    static void tournament_select (const bool F_SELF_SELECTION,
                                   const uint32_t SELECTION_SIZE,
                                   const TFloat SELECTION_P,
                                   const uint32_t NUM_ISLES,
                                   const uint32_t NUM_AGENTS,
                                   const TFloat * const fitness_array,
                                   uint32_t * selection_array,
                                   const TFloat * const prnumbers_array);

    static void whole_crossover (const TFloat CROSSOVER_RATE,
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

    static void migration_ring (const uint32_t NUM_ISLES,
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
#include "ga_operators_cpu.cpp"
#endif
