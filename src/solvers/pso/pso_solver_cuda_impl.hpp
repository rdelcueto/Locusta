#include "pso_solver_cuda.hpp"

namespace locusta {

    ///Interface for Genetic Algorithm solvers
    template<typename TFloat>
    pso_solver_cuda<TFloat>::pso_solver_cuda(population_set_cuda<TFloat> * population,
                                             evaluator_cuda<TFloat> * evaluator,
                                             prngenerator_cuda<TFloat> * prn_generator,
                                             uint32_t generation_target,
                                             TFloat * upper_bounds,
                                             TFloat * lower_bounds)

        : evolutionary_solver_cuda<TFloat>(population,
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
        CudaSafeCall(cudaMalloc((void **) &(_dev_bulk_prnumbers), _bulk_size * sizeof(TFloat)));

        // Allocate PSO resources
        const size_t TOTAL_GENES = _population->_TOTAL_GENES;

        CudaSafeCall(cudaMalloc((void **) &(_dev_cognitive_position_vector), TOTAL_GENES * sizeof(TFloat)));
        CudaSafeCall(cudaMalloc((void **) &(_dev_velocity_vector), TOTAL_GENES * sizeof(TFloat)));

        CudaCheckError();
    }

    template<typename TFloat>
    pso_solver_cuda<TFloat>::~pso_solver_cuda()
    {
        CudaSafeCall(cudaFree(_dev_cognitive_position_vector));
        CudaSafeCall(cudaFree(_dev_velocity_vector));
        CudaCheckError();
    }

    template<typename TFloat>
    void pso_solver_cuda<TFloat>::setup_solver()
    {
        evolutionary_solver_cuda<TFloat>::setup_solver();

        // Initialize best particle position with random positions.
        evolutionary_solver_cuda<TFloat>::initialize_vector(_dev_cognitive_position_vector, _dev_velocity_vector);

        // TODO: Set fill_velocities_kernel dispatch
    }

    template<typename TFloat>
    void pso_solver_cuda<TFloat>::teardown_solver()
    {

    }

    template<typename TFloat>
    void pso_solver_cuda<TFloat>::setup_operators(UpdateSpeedCudaFunctor<TFloat> * speed_functor_ptr,
                                                  UpdatePositionCudaFunctor<TFloat> * position_functor_ptr)
    {
        _speed_updater_ptr = speed_functor_ptr;
        _position_updater_ptr = position_functor_ptr;
    }

    template<typename TFloat>
    void pso_solver_cuda<TFloat>::set_migration_config(uint32_t migration_step,
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
    void pso_solver_cuda<TFloat>::transform()
    {
        (*_speed_updater_ptr)(this);
        (*_position_updater_ptr)(this);
    }

} // namespace locusta
