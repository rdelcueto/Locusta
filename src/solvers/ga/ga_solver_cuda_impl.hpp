#include "ga_solver_cuda.hpp"

namespace locusta {

template <typename TFloat>
void elite_population_replace_dispatch(
  const uint32_t ISLES, const uint32_t AGENTS, const uint32_t DIMENSIONS,
  const uint32_t* min_agent_idx, const TFloat* max_agent_genome,
  const TFloat* max_agent_fitness, TFloat* genomes, TFloat* fitness);

/// Interface for Genetic Algorithm solvers
template <typename TFloat>
ga_solver_cuda<TFloat>::ga_solver_cuda(population_set_cuda<TFloat>* population,
                                       evaluator_cuda<TFloat>* evaluator,
                                       prngenerator_cuda<TFloat>* prn_generator,
                                       uint64_t generation_target,
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
  _mutation_rate = 0.1;

  // Allocate GA resources
  const size_t TOTAL_AGENTS = _population->_TOTAL_AGENTS;

  CudaSafeCall(cudaMalloc((void**)&(_dev_couples_idx_array),
                          TOTAL_AGENTS * sizeof(uint32_t)));
  CudaSafeCall(cudaMalloc((void**)&(_dev_candidates_reservoir_array),
                          _ISLES * _AGENTS * _AGENTS * sizeof(uint32_t)));
}

template <typename TFloat>
ga_solver_cuda<TFloat>::~ga_solver_cuda()
{
  CudaSafeCall(cudaFree(_dev_couples_idx_array));
  CudaSafeCall(cudaFree(_dev_candidates_reservoir_array));
}

template <typename TFloat>
void
ga_solver_cuda<TFloat>::setup_solver()
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
ga_solver_cuda<TFloat>::teardown_solver()
{
  delete[] _prn_sets;
  CudaSafeCall(cudaFree(_dev_bulk_prns));
}

template <typename TFloat>
void
ga_solver_cuda<TFloat>::setup_operators(
  BreedCudaFunctor<TFloat>* breed_functor_ptr,
  SelectionCudaFunctor<TFloat>* selection_functor_ptr)
{
  _breed_functor_ptr = breed_functor_ptr;
  _selection_functor_ptr = selection_functor_ptr;
}

template <typename TFloat>
void
ga_solver_cuda<TFloat>::solver_config(uint32_t migration_step,
                                      uint32_t migration_size,
                                      uint32_t migration_selection_size,
                                      uint32_t selection_size,
                                      TFloat selection_stochastic_factor,
                                      TFloat crossover_rate,
                                      TFloat mutation_rate)
{
  _migration_step = migration_step;
  _migration_size = migration_size;
  _migration_selection_size = migration_selection_size;
  _selection_size = selection_size;
  _selection_stochastic_factor = selection_stochastic_factor;
  _crossover_rate = crossover_rate;
  _mutation_rate = mutation_rate;
}

template <typename TFloat>
void
ga_solver_cuda<TFloat>::transform()
{
  elite_population_replace();

  (*_selection_functor_ptr)(this);
  (*_breed_functor_ptr)(this);
}

template <typename TFloat>
void
ga_solver_cuda<TFloat>::elite_population_replace()
{

  TFloat* genomes = _dev_population->_dev_data_array;
  TFloat* fitness = _dev_population->_dev_fitness_array;

  elite_population_replace_dispatch(_ISLES, _AGENTS, _DIMENSIONS,
                                    _dev_min_agent_idx, _dev_max_agent_genome,
                                    _dev_max_agent_fitness, genomes, fitness);
}

} // namespace locusta
