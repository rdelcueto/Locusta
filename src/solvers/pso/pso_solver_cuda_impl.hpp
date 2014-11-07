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
                                           lower_bounds) {
        // Defaults
        _migration_step = 0;
        _migration_size = 1;
        _migration_selection_size = 2;
        _inertia_factor = 0.8;
        _cognitive_factor = 2.0;
        _social_factor = 2.0;

        // Allocate PSO resources
        const size_t TOTAL_GENES = _population->_TOTAL_GENES;
        const size_t TOTAL_AGENTS = _population->_TOTAL_AGENTS;

        CudaSafeCall(cudaMalloc((void **) &(_dev_cognitive_position_vector), TOTAL_GENES * sizeof(TFloat)));
        CudaSafeCall(cudaMalloc((void **) &(_dev_cognitive_fitness_vector), TOTAL_AGENTS * sizeof(TFloat)));
        CudaSafeCall(cudaMalloc((void **) &(_dev_velocity_vector), TOTAL_GENES * sizeof(TFloat)));
    }

    template<typename TFloat>
    pso_solver_cuda<TFloat>::~pso_solver_cuda() {
        CudaSafeCall(cudaFree(_dev_cognitive_position_vector));
        CudaSafeCall(cudaFree(_dev_cognitive_fitness_vector));
        CudaSafeCall(cudaFree(_dev_velocity_vector));
    }

    template<typename TFloat>
    void pso_solver_cuda<TFloat>::setup_solver() {
         // Pseudo random number allocation.
        const uint32_t RECORD_UPDATE_OFFSET = _particle_record_updater_ptr->required_prns(this);
        const uint32_t SPEED_UPDATE_OFFSET = _speed_updater_ptr->required_prns(this);
        const uint32_t POSITION_UPDATE_OFFSET = _position_updater_ptr->required_prns(this);

        _bulk_size = RECORD_UPDATE_OFFSET + SPEED_UPDATE_OFFSET + POSITION_UPDATE_SET;
        CudaSafeCall(cudaMalloc((void **) &(_dev_bulk_prns), _bulk_size * sizeof(TFloat)));

        _prn_sets = new TFloat*[3];
        _prn_sets[RECORD_UPDATE_SET] = _dev_bulk_prns;
        _prn_sets[SPEED_UPDATE_SET] = _dev_bulk_prns + RECORD_UPDATE_OFFSET;
        _prn_sets[POSITION_UPDATE_SET] = _dev_bulk_prns + RECORD_UPDATE_OFFSET + SPEED_UPDATE_OFFSET;

        TFloat * temporal_data = _dev_population->_dev_transformed_data_array;
        TFloat * temporal_data_fitness = _dev_population->_dev_fitness_array;

        const uint32_t TOTAL_GENES = _population->_TOTAL_GENES;
        const uint32_t TOTAL_AGENTS = _population->_TOTAL_AGENTS;

        // Initialize best particle position with random positions.
        evolutionary_solver_cuda<TFloat>::initialize_vector(_dev_cognitive_position_vector);

        // Copy data values into temporal vector.
        CudaSafeCall(cudaMemcpy(temporal_data,
                                _dev_cognitive_position_vector,
                                TOTAL_GENES * sizeof(TFloat),
                                cudaMemcpyDeviceToDevice));

        // Evaluate cognitive vector fitness.
        _population->swap_data_sets();
        evolutionary_solver<TFloat>::evaluate_genomes();
        _population->swap_data_sets();

        // Copy evaluation values.
        CudaSafeCall(cudaMemcpy(_dev_cognitive_fitness_vector,
                                temporal_data_fitness,
                                TOTAL_AGENTS * sizeof(TFloat),
                                cudaMemcpyDeviceToDevice));

        // TODO: Fill array with zeroes
        //CudaSafeCall(cudaMemset(_dev_velocity_vector, 0, TOTAL_GENES * sizeof(TFloat)));

        evolutionary_solver_cuda<TFloat>::setup_solver();
    }

    template<typename TFloat>
    void pso_solver_cuda<TFloat>::teardown_solver() {
        delete [] _prn_sets;
        CudaSafeCall(cudaFree(_dev_bulk_prns));
    }

    template<typename TFloat>
    void pso_solver_cuda<TFloat>::setup_operators(UpdateParticleRecordCudaFunctor<TFloat> * update_particle_record_functor_ptr,
                                                  UpdateSpeedCudaFunctor<TFloat> * update_speed_functor_ptr,
                                                  UpdatePositionCudaFunctor<TFloat> * update_position_functor_ptr) {
        _particle_record_updater_ptr = update_particle_record_functor_ptr;
        _speed_updater_ptr = update_speed_functor_ptr;
        _position_updater_ptr = update_position_functor_ptr;
    }

    template<typename TFloat>
    void pso_solver_cuda<TFloat>::solver_config(uint32_t migration_step,
                                                uint32_t migration_size,
                                                uint32_t migration_selection_size,
                                                TFloat inertia_factor,
                                                TFloat cognitive_factor,
                                                TFloat social_factor) {
        _migration_step = migration_step;
        _migration_size = migration_size;
        _migration_selection_size = migration_selection_size;
        _inertia_factor = inertia_factor;
        _cognitive_factor = cognitive_factor;
        _social_factor = social_factor;
    }

    template<typename TFloat>
    void pso_solver_cuda<TFloat>::transform() {
        (*_particle_record_updater_ptr)(this);
        (*_speed_updater_ptr)(this);
        (*_position_updater_ptr)(this);

        // Crop transformation vector
        evolutionary_solver_cuda<TFloat>::crop_vector(_dev_population->_dev_transformed_data_array);
    }

} // namespace locusta
