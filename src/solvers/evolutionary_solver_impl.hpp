#include "evolutionary_solver.hpp"

namespace locusta {

template <typename TFloat>
evolutionary_solver<TFloat>::evolutionary_solver(
  population_set<TFloat>* population, evaluator<TFloat>* evaluator,
  prngenerator<TFloat>* prn_generator, uint32_t generation_target,
  TFloat* upper_bounds, TFloat* lower_bounds)
  : _population(population)
  , _evaluator(evaluator)
  , _bulk_prn_generator(prn_generator)
  , _generation_target(generation_target)
  , _generation_count(0)
  , _ISLES(population->_ISLES)
  , _AGENTS(population->_AGENTS)
  , _DIMENSIONS(population->_DIMENSIONS)
  , _UPPER_BOUNDS(new TFloat[_DIMENSIONS])
  , _LOWER_BOUNDS(new TFloat[_DIMENSIONS])
  , _VAR_RANGES(new TFloat[_DIMENSIONS])
  , _max_agent_genome(new TFloat[_ISLES * _DIMENSIONS])
  , _min_agent_genome(new TFloat[_ISLES * _DIMENSIONS])
  , _max_agent_fitness(new TFloat[_ISLES])
  , _min_agent_fitness(new TFloat[_ISLES])
  , _max_agent_idx(new uint32_t[_ISLES])
  , _min_agent_idx(new uint32_t[_ISLES])
{

  for (uint32_t i = 0; i < _DIMENSIONS; i++) {
    _UPPER_BOUNDS[i] = upper_bounds[i];
    _LOWER_BOUNDS[i] = lower_bounds[i];
    _VAR_RANGES[i] = upper_bounds[i] - lower_bounds[i];
  }
}

template <typename TFloat>
evolutionary_solver<TFloat>::~evolutionary_solver()
{
  delete[] _UPPER_BOUNDS;
  delete[] _LOWER_BOUNDS;
  delete[] _VAR_RANGES;

  delete[] _max_agent_genome;
  delete[] _min_agent_genome;
  delete[] _max_agent_fitness;
  delete[] _min_agent_fitness;
  delete[] _max_agent_idx;
  delete[] _min_agent_idx;
}

template <typename TFloat>
void
evolutionary_solver<TFloat>::advance()
{
  transform();

  _population->swap_data_sets();

  evaluate_genomes();
  update_records();
  regenerate_prns();

  _generation_count++;
}

template <typename TFloat>
void
evolutionary_solver<TFloat>::run()
{
  do {
    // print_solutions();
    // print_population();
    // print_transformation_diff();
    advance();
  } while (_generation_count % _generation_target != 0);
  print_solutions();
}

template <typename TFloat>
void
evolutionary_solver<TFloat>::evaluate_genomes()
{
  _evaluator->evaluate(this);
}

template <typename TFloat>
void
evolutionary_solver<TFloat>::regenerate_prns()
{
  _bulk_prn_generator->_generate(_bulk_size, _bulk_prns);
}

} // namespace locusta
