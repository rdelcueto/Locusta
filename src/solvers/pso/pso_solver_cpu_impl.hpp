#include "pso_solver_cpu.hpp"

namespace locusta {

/// Interface for Genetic Algorithm solvers
template <typename TFloat>
pso_solver_cpu<TFloat>::pso_solver_cpu(population_set_cpu<TFloat>* population,
                                       evaluator_cpu<TFloat>* evaluator,
                                       prngenerator_cpu<TFloat>* prn_generator,
                                       uint64_t generation_target,
                                       TFloat* upper_bounds,
                                       TFloat* lower_bounds)
  : evolutionary_solver_cpu<TFloat>(population, evaluator, prn_generator,
                                    generation_target, upper_bounds,
                                    lower_bounds)
{

  // Defaults
  _migration_step = 0;
  _migration_size = 1;
  _migration_selection_size = 2;
  _inertia_factor = 0.5;
  _cognitive_factor = 2.0;
  _social_factor = 2.0;

  // Allocate PSO resources
  const size_t TOTAL_GENES = _population->_TOTAL_GENES;
  const size_t TOTAL_AGENTS = _population->_TOTAL_AGENTS;

  _cognitive_position_vector = new TFloat[TOTAL_GENES];
  _cognitive_fitness_vector = new TFloat[TOTAL_AGENTS];
  _velocity_vector = new TFloat[TOTAL_GENES];
}

template <typename TFloat>
pso_solver_cpu<TFloat>::~pso_solver_cpu()
{
  delete[] _cognitive_position_vector;
  delete[] _cognitive_fitness_vector;
  delete[] _velocity_vector;
}

template <typename TFloat>
void
pso_solver_cpu<TFloat>::setup_solver()
{
  // Pseudo random number allocation.
  const uint32_t RECORD_UPDATE_OFFSET =
    _particle_record_updater_ptr->required_prns(this);
  const uint32_t SPEED_UPDATE_OFFSET = _speed_updater_ptr->required_prns(this);
  const uint32_t POSITION_UPDATE_OFFSET =
    _position_updater_ptr->required_prns(this);

  _bulk_size = RECORD_UPDATE_OFFSET + SPEED_UPDATE_OFFSET + POSITION_UPDATE_SET;
  _bulk_prns = new TFloat[_bulk_size];

  _prn_sets = new TFloat*[3];
  _prn_sets[RECORD_UPDATE_SET] = _bulk_prns;
  _prn_sets[SPEED_UPDATE_SET] = _bulk_prns + RECORD_UPDATE_OFFSET;
  _prn_sets[POSITION_UPDATE_SET] =
    _bulk_prns + RECORD_UPDATE_OFFSET + SPEED_UPDATE_OFFSET;

  // Initialize best particle position with random positions.
  TFloat* temporal_data = _population->_transformed_data_array;
  TFloat* temporal_data_fitness = _population->_fitness_array;

  const uint32_t TOTAL_GENES = _population->_TOTAL_GENES;
  const uint32_t TOTAL_AGENTS = _population->_TOTAL_AGENTS;

  evolutionary_solver_cpu<TFloat>::initialize_vector(
    _cognitive_position_vector);

  // Copy data values into temporal population.
  memcpy(temporal_data, _cognitive_position_vector,
         TOTAL_GENES * sizeof(TFloat));

  // Evaluate cognitive vector fitness.
  _population->swap_data_sets();
  evolutionary_solver<TFloat>::evaluate_genomes();
  _population->swap_data_sets();

  // Copy evaluation values.
  memcpy(_cognitive_fitness_vector, temporal_data_fitness,
         TOTAL_AGENTS * sizeof(TFloat));

  const TFloat zero_value = 0;
  // Initialize Velocity to 0
  std::fill(_velocity_vector, _velocity_vector + TOTAL_GENES, zero_value);

  evolutionary_solver_cpu<TFloat>::setup_solver();
}

template <typename TFloat>
void
pso_solver_cpu<TFloat>::teardown_solver()
{
  delete[] _prn_sets;
  delete[] _bulk_prns;
}

template <typename TFloat>
void
pso_solver_cpu<TFloat>::setup_operators(
  UpdateParticleRecordFunctor<TFloat>* update_particle_record_functor_ptr,
  UpdateSpeedFunctor<TFloat>* update_speed_functor_ptr,
  UpdatePositionFunctor<TFloat>* update_position_functor_ptr)
{
  _particle_record_updater_ptr = update_particle_record_functor_ptr;
  _speed_updater_ptr = update_speed_functor_ptr;
  _position_updater_ptr = update_position_functor_ptr;
}

template <typename TFloat>
void
pso_solver_cpu<TFloat>::solver_config(uint32_t migration_step,
                                      uint32_t migration_size,
                                      uint32_t migration_selection_size,
                                      TFloat inertia_factor,
                                      TFloat cognitive_factor,
                                      TFloat social_factor)
{
  _migration_step = migration_step;
  _migration_size = migration_size;
  _migration_selection_size = migration_selection_size;
  _inertia_factor = inertia_factor;
  _cognitive_factor = cognitive_factor;
  _social_factor = social_factor;
}

template <typename TFloat>
void
pso_solver_cpu<TFloat>::transform()
{
  (*_particle_record_updater_ptr)(this);
  (*_speed_updater_ptr)(this);
  (*_position_updater_ptr)(this);

  // Crop transformation vector
  evolutionary_solver_cpu<TFloat>::crop_vector(
    _population->_transformed_data_array);
}

} // namespace locusta
