#ifndef LOCUSTA_GA_STD_OPERATORS_CUDA_H
#define LOCUSTA_GA_STD_OPERATORS_CUDA_H

#include "../../../prngenerator/prngenerator_cuda.hpp"
#include "ga_operators_cuda.hpp"

namespace locusta {

/**
 * @brief Dispatch function for the whole crossover operator.
 *
 * @param ISLES Number of isles in the population.
 * @param AGENTS Number of agents per isle.
 * @param DIMENSIONS Number of dimensions per agent.
 * @param DEVIATION Deviation factor for mutation.
 * @param CROSSOVER_RATE Crossover rate.
 * @param MUTATION_RATE Mutation rate.
 * @param DIST_LIMIT Distribution limit for mutation.
 * @param VAR_RANGES Array of ranges for the genes.
 * @param LOWER_BOUNDS Array of lower bounds for the genes.
 * @param UPPER_BOUNDS Array of upper bounds for the genes.
 * @param prn_array Array of pseudo-random numbers.
 * @param couple_selection Array of couple selections.
 * @param parent_genomes Array of parent genomes.
 * @param offspring_genomes Array of offspring genomes.
 * @param local_generator Pseudo-random number generator.
 */
template<typename TFloat>
void
whole_crossover_dispatch(const uint32_t ISLES,
                         const uint32_t AGENTS,
                         const uint32_t DIMENSIONS,
                         const TFloat DEVIATION,
                         const TFloat CROSSOVER_RATE,
                         const TFloat MUTATION_RATE,
                         const uint32_t DIST_LIMIT,
                         const TFloat* VAR_RANGES,
                         const TFloat* LOWER_BOUNDS,
                         const TFloat* UPPER_BOUNDS,
                         const TFloat* prn_array,
                         const uint32_t* couple_selection,
                         const TFloat* parent_genomes,
                         TFloat* offspring_genomes,
                         prngenerator_cuda<TFloat>* local_generator);

/**
 * @brief Dispatch function for the tournament selection operator.
 *
 * @param ISLES Number of isles in the population.
 * @param AGENTS Number of agents per isle.
 * @param SELECTION_SIZE Tournament selection size.
 * @param SELECTION_P Selection stochastic factor.
 * @param fitness_array Array of fitness values.
 * @param prn_array Array of pseudo-random numbers.
 * @param couple_idx_array Array of couple indices.
 * @param candidates_reservoir_array Array of candidate reservoirs.
 */
template<typename TFloat>
void
tournament_selection_dispatch(const uint32_t ISLES,
                              const uint32_t AGENTS,
                              const uint32_t SELECTION_SIZE,
                              const TFloat SELECTION_P,
                              const TFloat* fitness_array,
                              const TFloat* prn_array,
                              uint32_t* couple_idx_array,
                              uint32_t* candidates_reservoir_array);

/**
 * @brief CUDA implementation of the whole crossover operator.
 *
 * This class implements the whole crossover operator for the CUDA architecture.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct WholeCrossoverCuda : BreedCudaFunctor<TFloat>
{

  const TFloat DEVIATION = 0.2;
  const uint32_t DIST_LIMIT = 3;

  /**
   * @brief Get the number of pseudo-random numbers required by the operator.
   *
   * @param solver Genetic algorithm solver.
   * @return Number of pseudo-random numbers required.
   */
  uint32_t required_prns(ga_solver_cuda<TFloat>* solver)
  {
    const uint32_t ISLES = solver->_ISLES;
    const uint32_t AGENTS = solver->_AGENTS;
    const uint32_t DIMENSIONS = solver->_DIMENSIONS;

    return ISLES * (AGENTS * (1 + DIMENSIONS));
  }

  /**
   * @brief Apply the breeding operator.
   *
   * @param solver Genetic algorithm solver.
   */
  void operator()(ga_solver_cuda<TFloat>* solver)
  {
    const uint32_t ISLES = solver->_ISLES;
    const uint32_t AGENTS = solver->_AGENTS;
    const uint32_t DIMENSIONS = solver->_DIMENSIONS;
    const TFloat* VAR_RANGES = solver->_DEV_VAR_RANGES;
    const TFloat* LOWER_BOUNDS = solver->_DEV_LOWER_BOUNDS;
    const TFloat* UPPER_BOUNDS = solver->_DEV_UPPER_BOUNDS;

    const TFloat* prn_array = const_cast<TFloat*>(
      solver->_prn_sets[ga_solver_cuda<TFloat>::BREEDING_SET]);

    const TFloat CROSSOVER_RATE = solver->_crossover_rate;
    const TFloat MUTATION_RATE = solver->_mutation_rate;

    const uint32_t* couple_selection =
      const_cast<uint32_t*>(solver->_dev_couples_idx_array);

    const TFloat* parent_genomes =
      const_cast<TFloat*>(solver->_dev_population->_dev_data_array);

    TFloat* offspring_genomes =
      solver->_dev_population->_dev_transformed_data_array;

    prngenerator_cuda<TFloat>* local_generator =
      solver->_dev_bulk_prn_generator;

    whole_crossover_dispatch(ISLES,
                             AGENTS,
                             DIMENSIONS,
                             DEVIATION,
                             CROSSOVER_RATE,
                             MUTATION_RATE,
                             DIST_LIMIT,
                             VAR_RANGES,
                             LOWER_BOUNDS,
                             UPPER_BOUNDS,
                             prn_array,
                             couple_selection,
                             parent_genomes,
                             offspring_genomes,
                             local_generator);
  }
};

/**
 * @brief CUDA implementation of the tournament selection operator.
 *
 * This class implements the tournament selection operator for the CUDA
 * architecture.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct TournamentSelectionCuda : SelectionCudaFunctor<TFloat>
{

  /**
   * @brief Get the number of pseudo-random numbers required by the operator.
   *
   * @param solver Genetic algorithm solver.
   * @return Number of pseudo-random numbers required.
   */
  uint32_t required_prns(ga_solver_cuda<TFloat>* solver)
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
   * @param solver Genetic algorithm solver.
   */
  void operator()(ga_solver_cuda<TFloat>* solver)
  {
    const uint32_t ISLES = solver->_ISLES;
    const uint32_t AGENTS = solver->_AGENTS;

    const uint32_t SELECTION_SIZE = solver->_selection_size;
    const TFloat SELECTION_P = solver->_selection_stochastic_factor;

    const TFloat* prn_array = const_cast<TFloat*>(
      solver->_prn_sets[ga_solver_cuda<TFloat>::SELECTION_SET]);

    const TFloat* fitness_array =
      const_cast<TFloat*>(solver->_dev_population->_dev_fitness_array);

    uint32_t* couple_idx_array = solver->_dev_couples_idx_array;
    uint32_t* candidates_reservoir_array =
      solver->_dev_candidates_reservoir_array;

    tournament_selection_dispatch(ISLES,
                                  AGENTS,
                                  SELECTION_SIZE,
                                  SELECTION_P,
                                  fitness_array,
                                  prn_array,
                                  couple_idx_array,
                                  candidates_reservoir_array);
  }
};
}

#endif
