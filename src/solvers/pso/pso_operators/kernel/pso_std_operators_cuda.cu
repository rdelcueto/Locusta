#include "cuda_common/cuda_helpers.h"
#include "../pso_std_operators_cuda_impl.hpp"

namespace locusta {

    /// GPU Kernels Shared Memory Pointer.
    extern __shared__ int evaluator_shared_memory[];
    const uint32_t REPETITIONS = 1e2;

    template <typename TFloat>
    __global__
    void canonical_speed_update_kernel
    (const uint32_t DIMENSIONS,
     const TFloat inertia_factor,
     const TFloat cognitive_factor,
     const TFloat social_factor,
     const TFloat * __restrict__ positions,
     const TFloat * __restrict__ best_positions,
     const TFloat * __restrict__ isle_best_positions,
     const TFloat * __restrict__ prng_vector,
     TFloat * __restrict__ velocities)
    {
        const uint32_t i = blockIdx.x; // ISLE
        const uint32_t j = threadIdx.x; // AGENT

        const uint32_t ISLES = gridDim.x;
        const uint32_t AGENTS = blockDim.x;

        const uint32_t prng_offset = ISLES * AGENTS * DIMENSIONS;

        // Each thread iterates over a single particle.
        for(uint32_t k = 0; k < DIMENSIONS; k++) {
            const uint32_t locus_offset = k * ISLES * AGENTS;
            const uint32_t particle_gene_idx = locus_offset + i * AGENTS + j;
            const uint32_t isle_best_positions_idx = k * ISLES + i;

            const TFloat c_rnd = prng_vector[particle_gene_idx];
            const TFloat s_rnd = prng_vector[particle_gene_idx + prng_offset];

            // TODO: From Shared memory
            const TFloat p_g = isle_best_positions[isle_best_positions_idx];

            const TFloat p_i = best_positions[particle_gene_idx];
            const TFloat x_i = positions[particle_gene_idx];

            const TFloat v_i = velocities[particle_gene_idx];

            velocities[particle_gene_idx] = inertia_factor * v_i +
                cognitive_factor * c_rnd * (p_i - x_i) +
                social_factor * s_rnd * (p_g - x_i);
        }
    }

    template <typename TFloat>
    void canonical_speed_update_dispatch
    (const uint32_t ISLES,
     const uint32_t AGENTS,
     const uint32_t DIMENSIONS,
     const TFloat inertia_factor,
     const TFloat cognitive_factor,
     const TFloat social_factor,
     const TFloat * positions,
     const TFloat * best_positions,
     const TFloat * isle_best_positions,
     const TFloat * prng_vector,
     TFloat * velocities)
    {
        std::cout << "CUDA SPEED DISPATCH READY!" << std::endl;
        canonical_speed_update_kernel
                     <<<ISLES, AGENTS>>>
                     (DIMENSIONS,
                      inertia_factor,
                      cognitive_factor,
                      social_factor,
                      positions,
                      best_positions,
                      isle_best_positions,
                      prng_vector,
                      velocities);
        CudaCheckError();
    }

    // Template Specialization (float)
    template
    void canonical_speed_update_dispatch<float>
    (const uint32_t ISLES,
     const uint32_t AGENTS,
     const uint32_t DIMENSIONS,
     const float inertia_factor,
     const float cognitive_factor,
     const float social_factor,
     const float * positions,
     const float * best_positions,
     const float * isle_best_positions,
     const float * prng_vector,
     float * velocities);

    // Template Specialization (double)
    template
    void canonical_speed_update_dispatch<double>
    (const uint32_t ISLES,
     const uint32_t AGENTS,
     const uint32_t DIMENSIONS,
     const double inertia_factor,
     const double cognitive_factor,
     const double social_factor,
     const double * positions,
     const double * best_positions,
     const double * isle_best_positions,
     const double * prng_vector,
     double * velocities);

    template <typename TFloat>
    __global__
    void canonical_position_update_kernel
    (const uint32_t DIMENSIONS,
     const TFloat * __restrict__ velocities,
     const TFloat * positions,
     TFloat * __restrict__ new_positions)
    {
        const uint32_t i = blockIdx.x; // ISLE
        const uint32_t j = threadIdx.x; // AGENT

        const uint32_t ISLES = gridDim.x;
        const uint32_t AGENTS = blockDim.x;

        // Each thread iterates over a single particle.
        for(uint32_t k = 0; k < DIMENSIONS; k++) {
            const uint32_t locus_offset = k * ISLES * AGENTS;
            const uint32_t particle_gene_idx = locus_offset + i * AGENTS + j;

            const TFloat current_velocity = velocities[particle_gene_idx];
            const TFloat current_position = positions[particle_gene_idx];

            new_positions[particle_gene_idx] = current_position + current_velocity;
        }
    }

    template <typename TFloat>
    void canonical_position_update_dispatch
    (const uint32_t ISLES,
     const uint32_t AGENTS,
     const uint32_t DIMENSIONS,
     const TFloat * velocities,
     const TFloat * positions,
     TFloat * new_positions)
    {
        std::cout << "CUDA POSITION DISPATCH READY!" << std::endl;

        canonical_position_update_kernel
            <<<ISLES, AGENTS>>>
            (DIMENSIONS,
             velocities,
             positions,
             new_positions);

        CudaCheckError();
    }

    // Template Specialization (float)
    template
    void canonical_position_update_dispatch<float>
    (const uint32_t ISLES,
     const uint32_t AGENTS,
     const uint32_t DIMENSIONS,
     const float * velocities,
     const float * positions,
     float * new_positions);

    // Template Specialization (double)
    template
    void canonical_position_update_dispatch<double>
    (const uint32_t ISLES,
     const uint32_t AGENTS,
     const uint32_t DIMENSIONS,
     const double * velocities,
     const double * positions,
     double * new_positions);

}
