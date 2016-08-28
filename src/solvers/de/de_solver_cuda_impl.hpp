#include "de_solver_cuda.hpp"

namespace locusta {

template <typename TFloat>
void trial_vector_replace_dispatch(const uint32_t ISLES, const uint32_t AGENTS,
                                   const uint32_t DIMENSIONS,
                                   TFloat* previous_vectors,
                                   const TFloat* previous_fitness,
                                   const TFloat* trial_vectors,
                                   TFloat* trial_fitness);

/// Interface for Differential Evolution solvers
template <typename TFloat>
de_solver_cuda<TFloat>::de_solver_cuda(population_set_cuda<TFloat>* population,
                                       evaluator_cuda<TFloat>* evaluator,
                                       prngenerator_cuda<TFloat>* prn_generator,
                                       uint32_t generation_target,
                                       TFloat* upper_bounds,
                                       TFloat* lower_bounds)

  : evolutionary_solver_cuda<TFloat>(population, evaluator, prn_generator,
                                     generation_target, upper_bounds,
                                     lower_bounds)
{
  // Defaults
  _migration_step = 0;
  _migration_size = 1;
  _migration_selection_size = 2;
  _selection_size = 2;
  _selection_stochastic_factor = 0;
  _crossover_rate = 0.9;
  _differential_scale_factor = 0.5;

  // Allocate GA resources
  const size_t TOTAL_AGENTS = _population->_TOTAL_AGENTS;

  CudaSafeCall(cudaMalloc((void**)&(_dev_previous_fitness_array),
                          TOTAL_AGENTS * sizeof(TFloat)));
  CudaSafeCall(cudaMalloc((void**)&(_dev_recombination_idx_array),
                          TOTAL_AGENTS * sizeof(uint32_t)));
  CudaSafeCall(cudaMalloc((void**)&(_dev_recombination_reservoir_array),
                          _ISLES * _AGENTS * _AGENTS * sizeof(uint32_t)));
}

template <typename TFloat>
de_solver_cuda<TFloat>::~de_solver_cuda()
{
  CudaSafeCall(cudaFree(_dev_previous_fitness_array));
  CudaSafeCall(cudaFree(_dev_recombination_idx_array));
  CudaSafeCall(cudaFree(_dev_recombination_reservoir_array));
}

template <typename TFloat>
void
de_solver_cuda<TFloat>::setup_solver()
{
  // Pseudo random number allocation.
  const uint32_t SELECTION_OFFSET = _selection_functor_ptr->required_prns(this);
  const uint32_t BREEDING_OFFSET = _breed_functor_ptr->required_prns(this);

  _bulk_size = SELECTION_OFFSET + BREEDING_OFFSET;
  CudaSafeCall(
    cudaMalloc((void**)&(_dev_bulk_prns), _bulk_size * sizeof(TFloat)));

  _prn_sets = new TFloat*[2];
  _prn_sets[SELECTION_SET] = _dev_bulk_prns;
  _prn_sets[BREEDING_SET] = _dev_bulk_prns + SELECTION_OFFSET;

  evolutionary_solver_cuda<TFloat>::setup_solver();
}

template <typename TFloat>
void
de_solver_cuda<TFloat>::teardown_solver()
{
  delete[] _prn_sets;
  CudaSafeCall(cudaFree(_dev_bulk_prns));
}

template <typename TFloat>
void
de_solver_cuda<TFloat>::setup_operators(
  DeBreedCudaFunctor<TFloat>* breed_functor_ptr,
  DeSelectionCudaFunctor<TFloat>* selection_functor_ptr)
{
  _breed_functor_ptr = breed_functor_ptr;
  _selection_functor_ptr = selection_functor_ptr;
}

template <typename TFloat>
void
de_solver_cuda<TFloat>::solver_config(uint32_t migration_step,
                                      uint32_t migration_size,
                                      uint32_t migration_selection_size,
                                      uint32_t selection_size,
                                      TFloat selection_stochastic_factor,
                                      TFloat crossover_rate,
                                      TFloat differential_scale_factor)
{
  _migration_step = migration_step;
  _migration_size = migration_size;
  _migration_selection_size = migration_selection_size;
  _selection_size = selection_size;
  _selection_stochastic_factor = selection_stochastic_factor;
  _crossover_rate = crossover_rate;
  _differential_scale_factor = differential_scale_factor;
}

template <typename TFloat>
void
de_solver_cuda<TFloat>::advance()
{
  // Store previous fitness evaluation values.
  const size_t TOTAL_AGENTS = _population->_TOTAL_AGENTS;
  TFloat* dev_population_data_fitness = _dev_population->_dev_fitness_array;

  // Copy evaluation values.
  CudaSafeCall(
    cudaMemcpy(_dev_previous_fitness_array, dev_population_data_fitness,
               TOTAL_AGENTS * sizeof(TFloat), cudaMemcpyDeviceToDevice));

  transform();

  _dev_population->swap_data_sets();
  // Evaluate trial vectors
  evolutionary_solver_cuda<TFloat>::evaluate_genomes();
  // Replace target vectors with trial vectors, if better solutions found
  trial_vector_replace();
  // Restore target vectors
  _dev_population->swap_data_sets();

  evolutionary_solver_cuda<TFloat>::update_records();
  evolutionary_solver_cuda<TFloat>::regenerate_prns();

  _generation_count++;
}

template <typename TFloat>
void
de_solver_cuda<TFloat>::transform()
{

  (*_selection_functor_ptr)(this);
  (*_breed_functor_ptr)(this);
}

template <typename TFloat>
void
de_solver_cuda<TFloat>::trial_vector_replace()
{

  TFloat* previous_vectors = _dev_population->_dev_transformed_data_array;
  const TFloat* previous_fitness = _dev_previous_fitness_array;

  const TFloat* trial_vectors = _dev_population->_dev_data_array;
  TFloat* trial_fitness = _dev_population->_dev_fitness_array;

  trial_vector_replace_dispatch(_ISLES, _AGENTS, _DIMENSIONS, previous_vectors,
                                previous_fitness, trial_vectors, trial_fitness);
}

} // namespace locusta
