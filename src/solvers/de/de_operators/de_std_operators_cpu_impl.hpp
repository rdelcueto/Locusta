#ifndef LOCUSTA_DE_STD_OPERATORS_H
#define LOCUSTA_DE_STD_OPERATORS_H

#include "de_operators.hpp"
#include "prngenerator/prngenerator_cpu.hpp"

#define likely(x) __builtin_expect((x), 1)
#define unlikely(x) __builtin_expect((x), 0)

namespace locusta {

/**
 * @brief Whole crossover operator for differential evolution.
 *
 * This class implements the whole crossover operator for differential
 * evolution, which performs crossover between a target vector and a trial
 * vector.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct DeWholeCrossover : DeBreedFunctor<TFloat>
{

  /**
   * @brief Perform crossover between two genes.
   *
   * @param difference_a_vector First difference vector.
   * @param difference_b_vector Second difference vector.
   * @param base_vector Base vector.
   * @param DIFFERENTIAL_SCALE_FACTOR Differential scale factor.
   * @return Trial gene.
   */
  inline TFloat GeneCrossOver(const TFloat difference_a_vector,
                              const TFloat difference_b_vector,
                              const TFloat base_vector,
                              const TFloat DIFFERENTIAL_SCALE_FACTOR)
  {
    TFloat trial_gene = difference_a_vector - difference_b_vector;
    trial_gene *= DIFFERENTIAL_SCALE_FACTOR;
    trial_gene += base_vector;
    return trial_gene;
  }

  /**
   * @brief Crop a gene to fit within the bounds.
   *
   * @param trial_gene Trial gene to crop.
   * @param lower_bound Lower bound of the gene.
   * @param upper_bound Upper bound of the gene.
   * @return Cropped gene.
   */
  inline TFloat GeneCrop(const TFloat trial_gene,
                         const TFloat lower_bound,
                         const TFloat upper_bound)
  {
    TFloat cropped_gene = trial_gene;
    cropped_gene = trial_gene < lower_bound ? lower_bound : trial_gene;
    cropped_gene = trial_gene > upper_bound ? upper_bound : trial_gene;
    return cropped_gene;
  }

  /**
   * @brief Get the number of pseudo-random numbers required by the operator.
   *
   * @param solver Differential evolution solver.
   * @return Number of pseudo-random numbers required.
   */
  uint32_t required_prns(de_solver_cpu<TFloat>* solver)
  {
    const uint32_t ISLES = solver->_ISLES;
    const uint32_t AGENTS = solver->_AGENTS;
    const uint32_t DIMENSIONS = solver->_DIMENSIONS;
    const uint32_t GENOME_RND_OFFSET = 1 + DIMENSIONS;

    return ISLES * AGENTS * GENOME_RND_OFFSET;
  }

  /**
   * @brief Apply the breeding operator.
   *
   * @param solver Differential evolution solver.
   */
  void operator()(de_solver_cpu<TFloat>* solver)
  {
    const uint32_t ISLES = solver->_ISLES;
    const uint32_t AGENTS = solver->_AGENTS;
    const uint32_t DIMENSIONS = solver->_DIMENSIONS;
    const TFloat* VAR_RANGES = solver->_VAR_RANGES;

    const uint32_t GENOME_RND_OFFSET = 1 + DIMENSIONS;

    const TFloat* prn_array = const_cast<TFloat*>(
      solver->_prn_sets[de_solver_cpu<TFloat>::BREEDING_SET]);

    const TFloat CROSSOVER_RATE = solver->_crossover_rate;
    const TFloat DIFFERENTIAL_SCALE_FACTOR = solver->_differential_scale_factor;

    const TFloat* current_vectors =
      const_cast<TFloat*>(solver->_population->_data_array);
    TFloat* trial_vectors = solver->_population->_transformed_data_array;

    const uint32_t* trial_selection =
      const_cast<uint32_t*>(solver->_recombination_idx_array);

#pragma omp for collapse(2)
    for (uint32_t i = 0; i < ISLES; ++i) {
      for (uint32_t j = 0; j < AGENTS; ++j) {
        const uint32_t ISLE_OFFSET = AGENTS * DIMENSIONS;
        const uint32_t BASE_OFFSET = i * ISLE_OFFSET + j * DIMENSIONS;

        const TFloat* target_vector = current_vectors + BASE_OFFSET;
        TFloat* trial_vector = trial_vectors + BASE_OFFSET;

        const uint32_t DIFFERENCE_VECTOR_A_IDX =
          trial_selection[i * AGENTS + j];
        const uint32_t DIFFERENCE_VECTOR_B_IDX =
          trial_selection[i * AGENTS + j + 1];
        const uint32_t BASE_VECTOR_IDX = trial_selection[i * AGENTS + j + 2];

        const uint32_t DIFFERENCE_A_OFFSET =
          i * ISLE_OFFSET + DIFFERENCE_VECTOR_A_IDX * DIMENSIONS;
        const uint32_t DIFFERENCE_B_OFFSET =
          i * ISLE_OFFSET + DIFFERENCE_VECTOR_B_IDX * DIMENSIONS;
        const uint32_t BASE_VECTOR_OFFSET =
          i * ISLE_OFFSET + BASE_VECTOR_IDX * DIMENSIONS;

        const TFloat* agents_prns =
          prn_array + i * AGENTS * GENOME_RND_OFFSET + j * GENOME_RND_OFFSET;
        const bool FORCE_PARAMETER_COPY_FLAG = (*agents_prns++);

        // TODO: Profile case:
        //         Assume CROSSOVER operations for all genes/dimensions. (with
        //         SIMD, this should be very fast)
        //         Then only replace genes/dimensions with those in target
        //         vector, if the criteria wasn't met
        //         Since this case is the least probable, this implementation
        //         might help branch predictions performance,
        //         while taking advantage of SIMD operations
        const TFloat* difference_a_vector =
          current_vectors + DIFFERENCE_A_OFFSET;
        const TFloat* difference_b_vector =
          current_vectors + DIFFERENCE_B_OFFSET;
        const TFloat* base_vector = current_vectors + BASE_VECTOR_OFFSET;

#pragma omp simd
        for (uint32_t k = 0; k < DIMENSIONS; ++k) {
          const bool CROSSOVER_FLAG = (agents_prns[k]) >= CROSSOVER_RATE;
          const TFloat& lower_bound = solver->_LOWER_BOUNDS[k];
          const TFloat& upper_bound = solver->_UPPER_BOUNDS[k];

          TFloat trial_gene = GeneCrossOver(difference_a_vector[k],
                                            difference_b_vector[k],
                                            base_vector[k],
                                            DIFFERENTIAL_SCALE_FACTOR);

          if (unlikely((k != FORCE_PARAMETER_COPY_FLAG) && !CROSSOVER_FLAG)) {
            trial_vector[k] = target_vector[k];
          } else {
            trial_vector[k] = GeneCrop(trial_gene, lower_bound, upper_bound);
          }
        }
      }
    }
  }
};

/**
 * @brief Random selection operator for differential evolution.
 *
 * This class implements the random selection operator for differential
 * evolution, which selects random vectors from the population.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct DeRandomSelection : DeSelectionFunctor<TFloat>
{

  /**
   * @brief Get the number of pseudo-random numbers required by the operator.
   *
   * @param solver Differential evolution solver.
   * @return Number of pseudo-random numbers required.
   */
  uint32_t required_prns(de_solver_cpu<TFloat>* solver)
  {
    const uint32_t ISLES = solver->_ISLES;
    const uint32_t AGENTS = solver->_AGENTS;
    const uint32_t RANDOM_VECTORS = 3;

    return ISLES * AGENTS * (AGENTS - (1 + RANDOM_VECTORS));
  }

  /**
   * @brief Apply the selection operator.
   *
   * @param solver Differential evolution solver.
   */
  void operator()(de_solver_cpu<TFloat>* solver)
  {
    const uint32_t ISLES = solver->_ISLES;
    const uint32_t AGENTS = solver->_AGENTS;

    const uint32_t RANDOM_VECTORS = 3;

    const TFloat* prn_array = const_cast<TFloat*>(
      solver->_prn_sets[de_solver_cpu<TFloat>::SELECTION_SET]);
    const uint32_t GENOME_RND_OFFSET =
      ((AGENTS - (1 + RANDOM_VECTORS)) + (RANDOM_VECTORS - 1));

    uint32_t* recombination_idx_array = solver->_recombination_idx_array;

#pragma omp parallel for collapse(2)
    for (uint32_t i = 0; i < ISLES; ++i) {
      for (uint32_t j = 0; j < AGENTS; ++j) {
        const uint32_t ISLE_OFFSET = i * AGENTS;

        const TFloat* agents_prns = prn_array +
                                    (i * AGENTS * GENOME_RND_OFFSET) +
                                    (j * GENOME_RND_OFFSET);

        // Resevoir Sampling
        const uint32_t SAMPLE_SIZE = RANDOM_VECTORS;
        uint32_t candidates[SAMPLE_SIZE];

        for (uint32_t k = 0; k < (AGENTS - 1); ++k) {
          if (k < RANDOM_VECTORS) {
            // Fill
            candidates[k] = k < j ? k : k + 1;
          } else {
            uint32_t r;
            r = (*agents_prns++) * (k + 1);
            if (r < SAMPLE_SIZE) {
              // Replace
              candidates[r] = k < j ? k : k + 1;
            }
          }
        }

        const uint32_t idx = ISLE_OFFSET + j;
        for (uint32_t k = 0; k < RANDOM_VECTORS; ++k) {
          recombination_idx_array[idx + k] = candidates[k];
        }
      }
    }
  }
};

/**
 * @brief Tournament selection operator for differential evolution.
 *
 * This class implements the tournament selection operator for differential
 * evolution, which selects parents from a population based on their fitness.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct DeTournamentSelection : DeSelectionFunctor<TFloat>
{

  /**
   * @brief Get the number of pseudo-random numbers required by the operator.
   *
   * @param solver Differential evolution solver.
   * @return Number of pseudo-random numbers required.
   */
  uint32_t required_prns(de_solver_cpu<TFloat>* solver)
  {
    const uint32_t ISLES = solver->_ISLES;
    const uint32_t AGENTS = solver->_AGENTS;
    const uint32_t SELECTION_SIZE = solver->_selection_size;

    return ISLES * AGENTS *
           ((AGENTS - (1 + SELECTION_SIZE)) + (SELECTION_SIZE - 1));
  }

  /**
   * @brief Apply the selection operator.
   *
   * @param solver Differential evolution solver.
   */
  void operator()(de_solver_cpu<TFloat>* solver)
  {
    const uint32_t ISLES = solver->_ISLES;
    const uint32_t AGENTS = solver->_AGENTS;

    const uint32_t RANDOM_VECTORS = solver->_selection_size;
    const TFloat SELECTION_P = solver->_selection_stochastic_factor;

    const TFloat* prn_array = const_cast<TFloat*>(
      solver->_prn_sets[de_solver_cpu<TFloat>::SELECTION_SET]);
    const uint32_t RND_OFFSET =
      ((AGENTS - (1 + RANDOM_VECTORS)) + (RANDOM_VECTORS - 1));

    const TFloat* fitness_array =
      const_cast<TFloat*>(solver->_population->_fitness_array);

    uint32_t* recombination_idx_array = solver->_recombination_idx_array;

#pragma omp parallel for collapse(2)
    for (uint32_t i = 0; i < ISLES; ++i) {
      for (uint32_t j = 0; j < AGENTS; ++j) {
        const uint32_t ISLE_OFFSET = i * AGENTS;

        const TFloat* agents_prns =
          prn_array + (i * AGENTS * RND_OFFSET) + (j * RND_OFFSET);
        const uint32_t idx = ISLE_OFFSET + j;

        // Resevoir Sampling
        uint32_t candidates[RANDOM_VECTORS];
        // * Fill
        for (uint32_t k = 0; k < RANDOM_VECTORS; ++k) {
          candidates[k] = k < j ? k : k + 1;
        }

        // * Replace
        uint32_t selection_idx;
        const uint32_t iter_limit = AGENTS - 1;

        // TODO: Check prng cardinality.
        // AGENTS - (1 + RANDOM_VECTORS)

        for (uint32_t k = RANDOM_VECTORS; k < iter_limit; ++k) {
          selection_idx = (AGENTS - 1) * (*agents_prns);
          agents_prns++;

          if (selection_idx <= RANDOM_VECTORS) {
            candidates[selection_idx] = k < j ? k : k + 1;
          }
        }

        // Prefetch candidates fitness
        TFloat candidates_fitness[RANDOM_VECTORS];
        for (uint32_t k = 0; k < RANDOM_VECTORS; ++k) {
          candidates_fitness[k] = fitness_array[candidates[k] + ISLE_OFFSET];
        }

        // Tournament
        bool switch_flag;
        TFloat best_fitness = candidates_fitness[0];

        // TODO: Check prng cardinality.
        // RANDOM_VECTORS - 1

        for (uint32_t k = 1; k < RANDOM_VECTORS; ++k) {
          const TFloat candidate = candidates_fitness[k];
          switch_flag = (candidate > best_fitness);

          if ((SELECTION_P != 0.0f) && (SELECTION_P >= (*agents_prns))) {
            switch_flag = !switch_flag;
          }

          agents_prns++; // Advance pointer

          if (switch_flag) {
            best_fitness = candidate;
            candidates[0] = candidates[k];
          }
        }
        recombination_idx_array[idx] = candidates[0];
      }
    }
  }
};
}

#endif
