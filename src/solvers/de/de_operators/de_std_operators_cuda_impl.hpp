#ifndef LOCUSTA_DE_STD_OPERATORS_CUDA_H
#define LOCUSTA_DE_STD_OPERATORS_CUDA_H

#include "../../../prngenerator/prngenerator_cuda.hpp"
#include "de_operators_cuda.hpp"

namespace locusta {

/**
 * @brief Dispatch function for the whole crossover operator in differential
 * evolution.
 *
 * @param ISLES Number of isles in the population.
 * @param AGENTS Number of agents per isle.
 * @param DIMENSIONS Number of dimensions per agent.
 * @param CROSSOVER_RATE Crossover rate.
 * @param DIFFERENTIAL_SCALE_FACTOR Differential scale factor.
 * @param VAR_RANGES Array of ranges for the genes.
 * @param LOWER_BOUNDS Array of lower bounds for the genes.
 * @param UPPER_BOUNDS Array of upper bounds for the genes.
 * @param prn_array Array of pseudo-random numbers.
 * @param trial_selection Array of trial selections.
 * @param current_vectors Array of current vectors.
 * @param trial_vectors Array of trial vectors.
 * @param local_generator Pseudo-random number generator.
 */
template<typename TFloat>
void
de_whole_crossover_dispatch(const uint32_t ISLES,
                            const uint32_t AGENTS,
                            const uint32_t DIMENSIONS,
                            const TFloat CROSSOVER_RATE,
                            const TFloat DIFFERENTIAL_SCALE_FACTOR,
                            const TFloat* VAR_RANGES,
                            const TFloat* LOWER_BOUNDS,
                            const TFloat* UPPER_BOUNDS,
                            const TFloat* prn_array,
                            const uint32_t* trial_selection,
                            const TFloat* current_vectors,
                            TFloat* trial_vectors,
                            prngenerator_cuda<TFloat>* local_generator);

/**
 * @brief Dispatch function for the random selection operator in differential
 * evolution.
 *
 * @param ISLES Number of isles in the population.
 * @param AGENTS Number of agents per isle.
 * @param prn_array Array of pseudo-random numbers.
 * @param couple_idx_array Array of couple indices.
 * @param candidates_reservoir_array Array of candidate reservoirs.
 */
template<typename TFloat>
void
de_random_selection_dispatch(const uint32_t ISLES,
                             const uint32_t AGENTS,
                             const TFloat* prn_array,
                             uint32_t* couple_idx_array,
                             uint32_t* candidates_reservoir_array);

/**
 * @brief Dispatch function for the tournament selection operator in
 * differential evolution.
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
de_tournament_selection_dispatch(const uint32_t ISLES,
                                 const uint32_t AGENTS,
                                 const uint32_t SELECTION_SIZE,
                                 const TFloat SELECTION_P,
                                 const TFloat* fitness_array,
                                 const TFloat* prn_array,
                                 uint32_t* couple_idx_array,
                                 uint32_t* candidates_reservoir_array);

/**
 * @brief CUDA implementation of the whole crossover operator for differential
 * evolution.
 *
 * This class implements the whole crossover operator for differential evolution
 * on the CUDA architecture.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct DeWholeCrossoverCuda : DeBreedCudaFunctor<TFloat>
{

  /**
   * @brief Get the number of pseudo-random numbers required by the operator.
   *
   * @param solver Differential evolution solver.
   * @return Number of pseudo-random numbers required.
   */
  uint32_t required_prns(de_solver_cuda<TFloat>* solver)
  {
    const uint32_t ISLES = solver->_ISLES;
    const uint32_t AGENTS = solver->_AGENTS;
    const uint32_t DIMENSIONS = solver->_DIMENSIONS;

    return ISLES * AGENTS * (1 + DIMENSIONS);
  }

  /**
   * @brief Apply the breeding operator.
   *
   * @param solver Differential evolution solver.
   */
  void operator()(de_solver_cuda<TFloat>* solver)
  {
    const uint32_t ISLES = solver->_ISLES;
    const uint32_t AGENTS = solver->_AGENTS;
    const uint32_t DIMENSIONS = solver->_DIMENSIONS;
    const TFloat* VAR_RANGES = solver->_DEV_VAR_RANGES;
    const TFloat* LOWER_BOUNDS = solver->_DEV_LOWER_BOUNDS;
    const TFloat* UPPER_BOUNDS = solver->_DEV_UPPER_BOUNDS;

    const TFloat DEVIATION = 0.2;

    const TFloat* prn_array = const_cast<TFloat*>(
      solver->_prn_sets[de_solver_cuda<TFloat>::BREEDING_SET]);

    const TFloat CROSSOVER_RATE = solver->_crossover_rate;
    const TFloat DIFFERENTIAL_SCALE_FACTOR = solver->_differential_scale_factor;

    const uint32_t* trial_selection =
      const_cast<uint32_t*>(solver->_dev_recombination_idx_array);

    const TFloat* current_vectors =
      const_cast<TFloat*>(solver->_dev_population->_dev_data_array);

    TFloat* trial_vectors =
      solver->_dev_population->_dev_transformed_data_array;

    prngenerator_cuda<TFloat>* local_generator =
      solver->_dev_bulk_prn_generator;

    de_whole_crossover_dispatch(ISLES,
                                AGENTS,
                                DIMENSIONS,
                                CROSSOVER_RATE,
                                DIFFERENTIAL_SCALE_FACTOR,
                                VAR_RANGES,
                                LOWER_BOUNDS,
                                UPPER_BOUNDS,
                                prn_array,
                                trial_selection,
                                current_vectors,
                                trial_vectors,
                                local_generator);
  }
};

/**
 * @brief CUDA implementation of the random selection operator for differential
 * evolution.
 *
 * This class implements the random selection operator for differential
 * evolution on the CUDA architecture.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct DeRandomSelectionCuda : DeSelectionCudaFunctor<TFloat>
{

  /**
   * @brief Get the number of pseudo-random numbers required by the operator.
   *
   * @param solver Differential evolution solver.
   * @return Number of pseudo-random numbers required.
   */
  uint32_t required_prns(de_solver_cuda<TFloat>* solver)
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
  void operator()(de_solver_cuda<TFloat>* solver)
  {
    const uint32_t ISLES = solver->_ISLES;
    const uint32_t AGENTS = solver->_AGENTS;

    const TFloat* prn_array = const_cast<TFloat*>(
      solver->_prn_sets[de_solver_cuda<TFloat>::SELECTION_SET]);

    uint32_t* recombination_idx_array = solver->_dev_recombination_idx_array;
    uint32_t* recombination_reservoir_array =
      solver->_dev_recombination_reservoir_array;

    de_random_selection_dispatch(ISLES,
                                 AGENTS,
                                 prn_array,
                                 recombination_idx_array,
                                 recombination_reservoir_array);
  }
};

/**
 * @brief CUDA implementation of the tournament selection operator for
 * differential evolution.
 *
 * This class implements the tournament selection operator for differential
 * evolution on the CUDA architecture.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct DeTournamentSelectionCuda : DeSelectionCudaFunctor<TFloat>
{

  /**
   * @brief Get the number of pseudo-random numbers required by the operator.
   *
   * @param solver Differential evolution solver.
   * @return Number of pseudo-random numbers required.
   */
  uint32_t required_prns(de_solver_cuda<TFloat>* solver)
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
  void operator()(de_solver_cuda<TFloat>* solver)
  {
    const uint32_t ISLES = solver->_ISLES;
    const uint32_t AGENTS = solver->_AGENTS;

    const uint32_t SELECTION_SIZE = solver->_selection_size;
    const TFloat SELECTION_P = solver->_selection_stochastic_factor;

    const TFloat* prn_array = const_cast<TFloat*>(
      solver->_prn_sets[de_solver_cuda<TFloat>::SELECTION_SET]);

    const TFloat* fitness_array =
      const_cast<TFloat*>(solver->_dev_population->_dev_fitness_array);

    uint32_t* recombination_idx_array = solver->_dev_recombination_idx_array;
    uint32_t* recombination_reservoir_array =
      solver->_dev_recombination_reservoir_array;

    de_tournament_selection_dispatch(ISLES,
                                     AGENTS,
                                     SELECTION_SIZE,
                                     SELECTION_P,
                                     fitness_array,
                                     prn_array,
                                     recombination_idx_array,
                                     recombination_reservoir_array);
  }
};
}

#endif
