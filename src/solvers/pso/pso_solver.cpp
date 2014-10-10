#include "pso_solver.hpp"

namespace locusta {

    ///Interface for Genetic Algorithm solvers
    template<typename TFloat>
    pso_solver<TFloat>::pso_solver(population_set<TFloat> * population,
                                   evaluator<TFloat> * evaluator,
                                   prngenerator<TFloat> * prn_generator,
                                   uint32_t generation_target,
                                   TFloat * upper_bounds,
                                   TFloat * lower_bounds)

        : evolutionary_solver<TFloat>(population,
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
        _inertia_factor = 0.9;
        _cognitive_factor = 1.0;
        _social_factor = 1.0;

        _bulk_size = 2 * _population->_TOTAL_GENES;
        // Memory allocation
        _bulk_prnumbers = new TFloat[_bulk_size];

        // Allocate PSO resources
        const size_t TOTAL_GENES = _population->_TOTAL_GENES;

        _cognitive_position_vector = new TFloat[TOTAL_GENES];
        _velocity_vector = new TFloat[TOTAL_GENES];
    }

    template<typename TFloat>
    pso_solver<TFloat>::~pso_solver()
    {
        delete [] _bulk_prnumbers;
        delete [] _cognitive_position_vector;
        delete [] _velocity_vector;
    }

    template<typename TFloat>
    void pso_solver<TFloat>::setup_solver()
    {
        if( !_population->_f_initialized ) {
           evolutionary_solver<TFloat>::initialize_population();
        }

        if( _f_initialized ) {
            teardown_solver();
        }

        const size_t TOTAL_GENES = _population->_TOTAL_GENES;

        // Initialize best particle position with current position.
        memcpy(_cognitive_position_vector, _population->_data_array, TOTAL_GENES * sizeof(TFloat));

        // Initialize Velocity
        _bulk_prn_generator->_generate(TOTAL_GENES, _velocity_vector);
    }

    template<typename TFloat>
    void pso_solver<TFloat>::teardown_solver()
    {

    }

    template<typename TFloat>
    void pso_solver<TFloat>::setup_operators(UpdateSpeedFunctor<TFloat> * speed_functor_ptr,
                                             UpdatePositionFunctor<TFloat> * position_functor_ptr)
    {
        _speed_updater_ptr = speed_functor_ptr;
        _position_updater_ptr = position_functor_ptr;
    }

    template<typename TFloat>
    void pso_solver<TFloat>::set_migration_config(uint32_t migration_step,
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
    void pso_solver<TFloat>::transform()
    {
        (*_speed_updater_ptr)(this);
        (*_position_updater_ptr)(this);
    }

} // namespace locusta
