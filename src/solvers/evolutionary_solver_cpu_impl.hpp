#include "evolutionary_solver_cpu.hpp"

namespace locusta {

/**
 * @brief Construct a new evolutionary_solver_cpu object.
 *
 * @param population Population set.
 * @param evaluator Evaluator.
 * @param prn_generator Pseudo-random number generator.
 * @param generation_target Target number of generations.
 * @param upper_bounds Array of upper bounds for the genes.
 * @param lower_bounds Array of lower bounds for the genes.
 */
template<typename TFloat>
evolutionary_solver_cpu<TFloat>::evolutionary_solver_cpu(
  population_set_cpu<TFloat>* population,
  evaluator_cpu<TFloat>* evaluator,
  prngenerator_cpu<TFloat>* prn_generator,
  uint64_t generation_target,
  TFloat* upper_bounds,
  TFloat* lower_bounds)
  : evolutionary_solver<TFloat>(population,
                                evaluator,
                                prn_generator,
                                generation_target,
                                upper_bounds,
                                lower_bounds)
{
}

/**
 * @brief Destroy the evolutionary_solver_cpu object.
 */
template<typename TFloat>
evolutionary_solver_cpu<TFloat>::~evolutionary_solver_cpu()
{
}

/**
 * @brief Set up the solver.
 *
 * This method initializes and allocates the solver's runtime resources.
 */
template<typename TFloat>
void
evolutionary_solver_cpu<TFloat>::setup_solver()
{
  // Initialize Population
  if (!_population->_f_initialized) {
    initialize_vector(_population->_data_array);
    _population->_f_initialized = true;
  }

  // Initialize solver records
  const TFloat* data_array = const_cast<TFloat*>(_population->_data_array);
  for (uint32_t i = 0; i < _ISLES; ++i) {
    for (uint32_t j = 0; j < _AGENTS; ++j) {
      // Initialize best genomes
      _max_agent_idx[i] = 0;
      _min_agent_idx[i] = 0;

      _max_agent_fitness[i] = -std::numeric_limits<TFloat>::infinity();
      _min_agent_fitness[i] = std::numeric_limits<TFloat>::infinity();

      for (uint32_t k = 0; k < _DIMENSIONS; ++k) {
        const uint32_t data_idx =
          i * _AGENTS * _DIMENSIONS + 0 * _DIMENSIONS + k;

        _max_agent_genome[i * _DIMENSIONS + k] = data_array[data_idx];
        _min_agent_genome[i * _DIMENSIONS + k] = data_array[data_idx];
      }
    }
  }

  // Initialize fitness, evaluating initialization data.
  evolutionary_solver<TFloat>::evaluate_genomes();
  // Update solver records.
  update_records();
  // Initialize random numbers.
  evolutionary_solver<TFloat>::regenerate_prns();
}

/**
 * @brief Tear down the solver.
 *
 * This method terminates and deallocates the solver's runtime resources.
 */
template<typename TFloat>
void
evolutionary_solver_cpu<TFloat>::teardown_solver()
{
}

/**
 * @brief Update the best genomes records.
 *
 * This method updates the records of the best genomes found so far.
 */
template<typename TFloat>
void
evolutionary_solver_cpu<TFloat>::update_records()
{
  TFloat* genomes = _population->_data_array;
  TFloat* fitness = _population->_fitness_array;

  // Scan population
  for (uint32_t i = 0; i < _ISLES; i++) {

    uint32_t isle_max_idx = 0;
    TFloat isle_max_fitness = fitness[i * _AGENTS];
    uint32_t isle_min_idx = 0;
    TFloat isle_min_fitness = fitness[i * _AGENTS];

    for (uint32_t j = 1; j < _AGENTS; j++) {

      const TFloat candidate_fitness = fitness[i * _AGENTS + j];

      if (candidate_fitness > isle_max_fitness) {
        isle_max_fitness = candidate_fitness;
        isle_max_idx = j;
      }

      if (candidate_fitness < isle_min_fitness) {
        isle_min_fitness = candidate_fitness;
        isle_min_idx = j;
      }
    }

    // Update current isle records
    const TFloat curr_max = _max_agent_fitness[i];

    if (isle_max_fitness > curr_max) {
      _max_agent_idx[i] = isle_max_idx;
      _max_agent_fitness[i] = isle_max_fitness;

      TFloat* max_genome = _max_agent_genome + i * _DIMENSIONS;
      const uint32_t max_genomes_offset =
        i * _AGENTS * _DIMENSIONS + isle_max_idx * _DIMENSIONS;

      for (uint32_t k = 0; k < _DIMENSIONS; k++) {
        max_genome[k] = genomes[max_genomes_offset + k];
      }
    }

    const TFloat curr_min = _min_agent_fitness[i];

    if (isle_min_fitness < curr_min) {
      _min_agent_idx[i] = isle_min_idx;
      _min_agent_fitness[i] = isle_min_fitness;

      TFloat* min_genome = _min_agent_genome + i * _DIMENSIONS;
      const uint32_t min_genomes_offset =
        i * _AGENTS * _DIMENSIONS + isle_min_idx * _DIMENSIONS;

      for (uint32_t k = 0; k < _DIMENSIONS; k++) {
        min_genome[k] = genomes[min_genomes_offset + k];
      }
    }
  }
}

/**
 * @brief Crop a vector to fit within the bounds.
 *
 * This method crops the values of a vector to fit within the solver's bounds.
 *
 * @param vec Vector to crop.
 */
template<typename TFloat>
void
evolutionary_solver_cpu<TFloat>::crop_vector(TFloat* vec)
{
  // Initialize vector data, within given bounds.
  for (uint32_t i = 0; i < _ISLES; ++i) {
    for (uint32_t j = 0; j < _AGENTS; ++j) {
      for (uint32_t k = 0; k < _DIMENSIONS; ++k) {
        // Initialize Data Array
        const uint32_t data_idx =
          i * _AGENTS * _DIMENSIONS + j * _DIMENSIONS + k;

        // TODO: Flexible bound crop method
        const TFloat curr_value = vec[data_idx];
        const TFloat low_bound = _LOWER_BOUNDS[k];
        const TFloat high_bound = _UPPER_BOUNDS[k];

        TFloat crop_value = curr_value;

        crop_value = crop_value < low_bound ? low_bound : crop_value;
        crop_value = crop_value > high_bound ? high_bound : crop_value;

        // Crop
        if (curr_value != crop_value) {
          vec[data_idx] = crop_value;
        }
      }
    }
  }
}

/**
 * @brief Initialize a vector with uniform random values within the bounds.
 *
 * This method initializes a vector with uniform random values within the
 * solver's bounds.
 *
 * @param dst_vec Vector to initialize.
 */
template<typename TFloat>
void
evolutionary_solver_cpu<TFloat>::initialize_vector(TFloat* dst_vec)
{

  const uint32_t TOTAL_GENES = _population->_TOTAL_GENES;
  TFloat* tmp_vec = new TFloat[TOTAL_GENES];
  _bulk_prn_generator->_generate(TOTAL_GENES, tmp_vec);

  for (uint32_t i = 0; i < _ISLES; ++i) {
    for (uint32_t j = 0; j < _AGENTS; ++j) {
      for (uint32_t k = 0; k < _DIMENSIONS; ++k) {

        // Initialize Data Array
        const uint32_t data_idx =
          i * _AGENTS * _DIMENSIONS + j * _DIMENSIONS + k;

        dst_vec[data_idx] =
          _LOWER_BOUNDS[k] + (_VAR_RANGES[k] * tmp_vec[data_idx]);
      }
    }
  }

  delete[] tmp_vec;
}

/**
 * @brief Print the last transformation difference.
 *
 * This method prints the difference between the current population and the
 * previous population after the last transformation.
 */
template<typename TFloat>
void
evolutionary_solver_cpu<TFloat>::print_transformation_diff()
{
  TFloat* fitness = _population->_fitness_array;
  TFloat* genomes = _population->_data_array;
  TFloat* transformed_genomes = _population->_transformed_data_array;

  for (uint32_t i = 0; i < _ISLES; i++) {
    for (uint32_t j = 0; j < _AGENTS; j++) {
      std::cout << fitness[i * _AGENTS + j] << ", [" << i << ", " << j
                << "] : ";
      TFloat* a = genomes + i * _AGENTS * _DIMENSIONS + j * _DIMENSIONS;
      TFloat* b =
        transformed_genomes + i * _AGENTS * _DIMENSIONS + j * _DIMENSIONS;
      for (uint32_t k = 0; k < _DIMENSIONS; k++) {
        std::cout << (a[k] - b[k]);
        if (k == _DIMENSIONS - 1) {
          std::cout << "]\n";
        } else {
          std::cout << ", ";
        }
      }
    }
  }
}

/**
 * @brief Print the current population.
 *
 * This method prints all current genomes and their fitness.
 */
template<typename TFloat>
void
evolutionary_solver_cpu<TFloat>::print_population()
{
  TFloat* fitness = _population->_fitness_array;
  TFloat* genomes = _population->_data_array;

  for (uint32_t i = 0; i < _ISLES; i++) {
    for (uint32_t j = 0; j < _AGENTS; j++) {
      std::cout << fitness[i * _AGENTS + j] << ", [" << i << ", " << j
                << "] : ";
      TFloat* curr_genome =
        genomes + i * _AGENTS * _DIMENSIONS + j * _DIMENSIONS;
      for (uint32_t k = 0; k < _DIMENSIONS; k++) {
        std::cout << curr_genome[k];
        if (k == _DIMENSIONS - 1) {
          std::cout << "]\n";
        } else {
          std::cout << ", ";
        }
      }
    }
  }
}

/**
 * @brief Print the solver's current best found solutions.
 *
 * This method prints the solver's current best found solutions and their
 * fitness.
 */
template<typename TFloat>
void
evolutionary_solver_cpu<TFloat>::print_solutions()
{
  std::cout << "Solutions @ " << (_generation_count) + 1 << " / "
            << _generation_target << std::endl;
  for (uint32_t i = 0; i < _ISLES; i++) {
    std::cout << _max_agent_fitness[i] << " : [";
    for (uint32_t k = 0; k < _DIMENSIONS; k++) {
      std::cout << _max_agent_genome[i * _DIMENSIONS + k];
      if (k == _DIMENSIONS - 1) {
        std::cout << "]\n";
      } else {
        std::cout << ", ";
      }
    }
  }
}

} // namespace locusta
