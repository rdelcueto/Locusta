#include "pso_solver_cpu.hpp"

namespace locusta {

    ///Interface for Genetic Algorithm solvers
    template<typename TFloat>
    pso_solver_cpu<TFloat>::pso_solver_cpu(population_set_cpu<TFloat> * population,
                                           evaluator_cpu<TFloat> * evaluator,
                                           prngenerator_cpu<TFloat> * prn_generator,
                                           uint32_t generation_target,
                                           TFloat * upper_bounds,
                                           TFloat * lower_bounds)

        : evolutionary_solver_cpu<TFloat>(population,
                                          evaluator,
                                          prn_generator,
                                          generation_target,
                                          upper_bounds,
                                          lower_bounds)
    {
        // Defaults
        _migration_step = 0;
        _migration_size = 1;
        _migration_selection_size = 2;
        _inertia_factor = 0.8;
        _cognitive_factor = 2.0;
        _social_factor = 2.0;

        _bulk_size = 2 * _population->_TOTAL_GENES;
        // Memory allocation
        _bulk_prnumbers = new TFloat[_bulk_size];

        // Allocate PSO resources
        const size_t TOTAL_GENES = _population->_TOTAL_GENES;
        const size_t TOTAL_AGENTS = _population->_TOTAL_AGENTS;

        _cognitive_position_vector = new TFloat[TOTAL_GENES];
        _cognitive_fitness_vector = new TFloat[TOTAL_AGENTS];
        _velocity_vector = new TFloat[TOTAL_GENES];
    }

    template<typename TFloat>
    pso_solver_cpu<TFloat>::~pso_solver_cpu()
    {
        delete [] _bulk_prnumbers;
        delete [] _cognitive_position_vector;
        delete [] _cognitive_fitness_vector;
        delete [] _velocity_vector;
    }

    template<typename TFloat>
    void pso_solver_cpu<TFloat>::setup_solver()
    {
        // Initialize best particle position with random positions.
        TFloat * temporal_data = _population->_transformed_data_array;
        TFloat * temporal_data_fitness = _population->_fitness_array;

        const uint32_t TOTAL_GENES = _population->_TOTAL_GENES;
        const uint32_t TOTAL_AGENTS = _population->_TOTAL_AGENTS;

        evolutionary_solver_cpu<TFloat>::initialize_vector(temporal_data,
                                                           _velocity_vector);

        // Evaluate cognitive vector fitness.
        _population->swap_data_sets();
        evolutionary_solver<TFloat>::evaluate_genomes();
        _population->swap_data_sets();

        // Copy evaluation values.
        memcpy(_cognitive_fitness_vector,
               temporal_data_fitness,
               TOTAL_AGENTS * sizeof(TFloat));

        // Copy data values into cognitive vector.
        memcpy(_cognitive_position_vector,
               temporal_data,
               TOTAL_GENES * sizeof(TFloat));

        // Initialize Velocity to 0
        const size_t vec_size = _population->_TOTAL_GENES;
        std::fill(_velocity_vector, _velocity_vector + vec_size, 0);

        evolutionary_solver_cpu<TFloat>::setup_solver();
    }

    template<typename TFloat>
    void pso_solver_cpu<TFloat>::teardown_solver()
    {

    }

    template<typename TFloat>
    void pso_solver_cpu<TFloat>::setup_operators(UpdateParticleRecordFunctor<TFloat > * update_particle_record_functor_ptr,
                                                 UpdateSpeedFunctor<TFloat> * update_speed_functor_ptr,
                                                 UpdatePositionFunctor<TFloat> * update_position_functor_ptr)
    {
        _particle_record_updater_ptr = update_particle_record_functor_ptr;
        _speed_updater_ptr = update_speed_functor_ptr;
        _position_updater_ptr = update_position_functor_ptr;
    }

    template<typename TFloat>
    void pso_solver_cpu<TFloat>::set_migration_config(uint32_t migration_step,
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

    template<typename TFloat>
    void pso_solver_cpu<TFloat>::transform()
    {
        (*_particle_record_updater_ptr)(this);
        (*_speed_updater_ptr)(this);
        (*_position_updater_ptr)(this);
    }

} // namespace locusta
