#ifndef LOCUSTA_GA_STD_OPERATORS_H
#define LOCUSTA_GA_STD_OPERATORS_H

#include "ga_operators.hpp"
#include "prngenerator/prngenerator_cpu.hpp"

#ifndef likely
#define likely(x) __builtin_expect((x), 1)
#endif
#ifndef unlikely
#define unlikely(x) __builtin_expect((x), 0)
#endif

namespace locusta {

/**
 * @brief Whole crossover operator.
 *
 * This class implements the whole crossover operator, which performs crossover
 * between two parents to create an offspring.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct WholeCrossover : GaBreedFunctor<TFloat>
{

  const uint32_t DIST_LIMIT = 3;
  const TFloat INV_DIST_LIMIT = 1.0 / DIST_LIMIT;
  const TFloat DEVIATION = 0.2;

  /**
   * @brief Perform crossover between two genes.
   *
   * @param parent_gene_a First parent gene.
   * @param parent_gene_b Second parent gene.
   * @param GENE_CROSSOVER_FLAG Flag indicating whether to perform crossover.
   * @return Offspring gene.
   */
  inline TFloat GeneCrossOver(const TFloat parent_gene_a,
                              const TFloat parent_gene_b,
                              const TFloat GENE_CROSSOVER_FLAG)
  {

    TFloat offspring_gene = (0.5f * parent_gene_a) + (0.5f * parent_gene_b);

    if (unlikely(!GENE_CROSSOVER_FLAG)) {
      offspring_gene = parent_gene_a;
    }

    return offspring_gene;
  }

  /**
   * @brief Mutate a gene.
   *
   * @param offspring_gene Offspring gene to mutate.
   * @param agents_prns Array of pseudo-random numbers.
   * @param gene_range Range of the gene.
   * @return Mutated gene.
   */
  inline TFloat GeneMutate(const TFloat offspring_gene,
                           const TFloat* agents_prns,
                           const TFloat gene_range)
  {
    TFloat x = 0.0;
    TFloat mutated_gene = offspring_gene;

    for (uint32_t n = 0; n < DIST_LIMIT; ++n) {
      x += *(agents_prns + n);
    }

    x *= INV_DIST_LIMIT;
    x -= 0.5;
    x *= DEVIATION * gene_range;

    mutated_gene += x;
    return mutated_gene;
  }

  /**
   * @brief Crop a gene to fit within the bounds.
   *
   * @param offspring_gene Offspring gene to crop.
   * @param lower_bound Lower bound of the gene.
   * @param upper_bound Upper bound of the gene.
   * @return Cropped gene.
   */
  inline TFloat GeneCrop(const TFloat offspring_gene,
                         const TFloat lower_bound,
                         const TFloat upper_bound)
  {

    TFloat cropped_value = offspring_gene;

    cropped_value = cropped_value < lower_bound ? lower_bound : cropped_value;
    cropped_value = cropped_value > upper_bound ? upper_bound : cropped_value;

    return cropped_value;
  }

  /**
   * @brief Get the number of pseudo-random numbers required by the operator.
   *
   * @param solver Genetic algorithm solver.
   * @return Number of pseudo-random numbers required.
   */
  uint32_t required_prns(ga_solver_cpu<TFloat>* solver)
  {
    const uint32_t ISLES = solver->_ISLES;
    const uint32_t AGENTS = solver->_AGENTS;
    const uint32_t DIMENSIONS = solver->_DIMENSIONS;
    const uint32_t GENOME_RND_OFFSET = (1 + 1 + DIST_LIMIT) * DIMENSIONS;

    return ISLES * AGENTS * GENOME_RND_OFFSET;
  }

  /**
   * @brief Apply the breeding operator.
   *
   * @param solver Genetic algorithm solver.
   */
  void operator()(ga_solver_cpu<TFloat>* solver)
  {
#pragma omp parallel default(none) shared(solver)
    {
      const uint32_t ISLES = solver->_ISLES;
      const uint32_t AGENTS = solver->_AGENTS;
      const uint32_t DIMENSIONS = solver->_DIMENSIONS;
      const TFloat* VAR_RANGES = solver->_VAR_RANGES;

      const uint32_t GENOME_RND_OFFSET = (1 + 1 + DIST_LIMIT) * DIMENSIONS;

      const TFloat* prn_array = const_cast<TFloat*>(
        solver->_prn_sets[ga_solver_cpu<TFloat>::BREEDING_SET]);

      const TFloat CROSSOVER_RATE = solver->_crossover_rate;
      const TFloat MUTATION_RATE = solver->_mutation_rate;

      const TFloat* parent_genomes =
        const_cast<TFloat*>(solver->_population->_data_array);

      TFloat* offspring_genomes = solver->_population->_transformed_data_array;

      const uint32_t* couple_selection =
        const_cast<uint32_t*>(solver->_couples_idx_array);

#pragma omp parallel for
      for (uint32_t i = 0; i < ISLES; ++i) {
        for (uint32_t j = 0; j < AGENTS; ++j) {
          const uint32_t ISLE_OFFSET = AGENTS * DIMENSIONS;
          const uint32_t BASE_IDX = i * ISLE_OFFSET + j * DIMENSIONS;
          const uint32_t COUPLE_IDX = couple_selection[i * AGENTS + j];
          const uint32_t COUPLE_BASE_IDX =
            i * ISLE_OFFSET + COUPLE_IDX * DIMENSIONS;

          const TFloat* agents_prns = prn_array +
                                      (i * AGENTS * GENOME_RND_OFFSET) +
                                      (j * GENOME_RND_OFFSET);

          TFloat* offspring = offspring_genomes + BASE_IDX;
          const TFloat* const parentA = parent_genomes + BASE_IDX;
          const TFloat* const parentB = parent_genomes + COUPLE_BASE_IDX;
          TFloat offspring_gene;

          // TODO: Profile case:
          //         Assume CROSSOVER operations for all genes/dimensions.
          //         (with
          //         SIMD, this should be very fast)
          for (uint32_t k = 0; k < DIMENSIONS; ++k) {

            const TFloat& range = VAR_RANGES[k];
            const TFloat& lower_bound = solver->_LOWER_BOUNDS[k];
            const TFloat& upper_bound = solver->_UPPER_BOUNDS[k];

            const bool GENE_CROSSOVER_FLAG = *(agents_prns++) < CROSSOVER_RATE;
            const bool GENE_MUTATE_FLAG = *(agents_prns++) < MUTATION_RATE;

            // Apply Crossover transformation
            offspring_gene =
              GeneCrossOver(parentA[k], parentB[k], GENE_CROSSOVER_FLAG);

            // Apply Mutation
            if (unlikely(GENE_MUTATE_FLAG)) {
              offspring_gene = GeneMutate(offspring_gene, agents_prns, range);
            }

            // Crop and save final gene
            offspring[k] = GeneCrop(offspring_gene, lower_bound, upper_bound);
          }
        }
      }
    }
  }
};

/**
 * @brief Tournament selection operator.
 *
 * This class implements the tournament selection operator, which selects
 * parents from a population based on their fitness.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct TournamentSelection : GaSelectionFunctor<TFloat>
{

  /**
   * @brief Get the number of pseudo-random numbers required by the operator.
   *
   * @param solver Genetic algorithm solver.
   * @return Number of pseudo-random numbers required.
   */
  uint32_t required_prns(ga_solver_cpu<TFloat>* solver)
  {
    const uint32_t ISLES = solver->_ISLES;
    const uint32_t AGENTS = solver->_AGENTS;
    const uint32_t SELECTION_SIZE = solver->_selection_size;
    const uint32_t GENOME_RND_OFFSET =
      ((AGENTS - (1 + SELECTION_SIZE)) + (SELECTION_SIZE - 1));

    return ISLES * AGENTS * GENOME_RND_OFFSET;
  }

  /**
   * @brief Apply the selection operator.
   *
   * @param solver Genetic algorithm solver.
   */
  void operator()(ga_solver_cpu<TFloat>* solver)
  {
    const uint32_t ISLES = solver->_ISLES;
    const uint32_t AGENTS = solver->_AGENTS;

    const uint32_t SELECTION_SIZE = solver->_selection_size;
    const TFloat SELECTION_P = solver->_selection_stochastic_factor;

    const uint32_t GENOME_RND_OFFSET =
      ((AGENTS - (1 + SELECTION_SIZE)) + (SELECTION_SIZE - 1));

    const TFloat* prn_array = const_cast<TFloat*>(
      solver->_prn_sets[ga_solver_cpu<TFloat>::SELECTION_SET]);

    const TFloat* fitness_array =
      const_cast<TFloat*>(solver->_population->_fitness_array);

    uint32_t* couple_idx_array = solver->_couples_idx_array;

#pragma omp parallel for collapse(2)
    for (uint32_t i = 0; i < ISLES; ++i) {
      for (uint32_t j = 0; j < AGENTS; ++j) {
        const uint32_t ISLE_OFFSET = i * AGENTS;
        const TFloat* agents_prns = prn_array +
                                    (i * AGENTS * GENOME_RND_OFFSET) +
                                    (j * GENOME_RND_OFFSET);
        const uint32_t idx = ISLE_OFFSET + j;

        // Resevoir Sampling
        uint32_t candidates[SELECTION_SIZE];

        for (uint32_t k = 0; k < (AGENTS - 1); ++k) {
          if (k < SELECTION_SIZE) {
            // Fill
            candidates[k] = k < j ? k : k + 1;
          } else {
            uint32_t r;
            r = (*agents_prns++) * (k + 1);
            if (r < SELECTION_SIZE) {
              // Replace
              candidates[r] = k < j ? k : k + 1;
            }
          }
        }

        // Prefetch candidates fitness
        TFloat candidates_fitness[SELECTION_SIZE];
        for (uint32_t k = 0; k < SELECTION_SIZE; ++k) {
          candidates_fitness[k] = fitness_array[candidates[k] + ISLE_OFFSET];
        }

        // Tournament
        bool switch_flag;
        TFloat best_fitness = candidates_fitness[0];

        // Prng cardinality:
        //   SELECTION_SIZE - 1
        for (uint32_t k = 1; k < SELECTION_SIZE; ++k) {
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
        couple_idx_array[idx] = candidates[0];
      }
    }
  }
};
}

#endif
