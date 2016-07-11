#include "cuda_common/cuda_helpers.h"
#include "../pso_std_operators_cuda_impl.hpp"

namespace locusta {

  /// GPU Kernels Shared Memory Pointer.
  extern __shared__ int pso_operators_shared_memory[];

  template <typename TFloat>
  __global__
  void canonical_particle_record_update_kernel
  (const uint32_t DIMENSIONS,
   const TFloat * __restrict__ positions,
   const TFloat * __restrict__ fitness,
   TFloat * __restrict__ record_positions,
   TFloat * __restrict__ record_fitness)
  {
    const uint32_t i = blockIdx.x; // ISLE
    const uint32_t j = threadIdx.x; // AGENT

    const uint32_t ISLES = gridDim.x;
    const uint32_t AGENTS = blockDim.x;

    const uint32_t locus_offset = i * AGENTS + j;

    const TFloat curr_fitness = fitness[locus_offset];
    const TFloat candidate_fitness = record_fitness[locus_offset];

    if (curr_fitness > candidate_fitness) {
      record_fitness[locus_offset] = curr_fitness;
      // Each thread iterates over a single particle.
      for(uint32_t k = 0; k < DIMENSIONS; k++) {
        const uint32_t particle_gene_idx = k * ISLES * AGENTS + locus_offset;
        record_positions[particle_gene_idx] = positions[particle_gene_idx];
      }
    }
  }

  template <typename TFloat>
  void canonical_particle_update_dispatch
  (const uint32_t ISLES,
   const uint32_t AGENTS,
   const uint32_t DIMENSIONS,
   const TFloat * positions,
   const TFloat * fitness,
   TFloat * record_positions,
   TFloat * record_fitness)
  {
    canonical_particle_record_update_kernel
      <<<ISLES, AGENTS>>>
      (DIMENSIONS,
       positions,
       fitness,
       record_positions,
       record_fitness);

    CudaCheckError();
  }

  // Template Specialization (float)
  template
  void canonical_particle_update_dispatch<float>
  (const uint32_t ISLES,
   const uint32_t AGENTS,
   const uint32_t DIMENSIONS,
   const float * positions,
   const float * fitness,
   float * record_positions,
   float * record_fitness);

  // Template Specialization (double)
  template
  void canonical_particle_update_dispatch<double>
  (const uint32_t ISLES,
   const uint32_t AGENTS,
   const uint32_t DIMENSIONS,
   const double * positions,
   const double * fitness,
   double * record_positions,
   double * record_fitness);

  template <typename TFloat>
  __global__
  void canonical_speed_update_kernel
  (const uint32_t DIMENSIONS,
   const TFloat inertia_factor,
   const TFloat cognitive_factor,
   const TFloat social_factor,
   const TFloat * __restrict__ positions,
   const TFloat * __restrict__ record_positions,
   const TFloat * __restrict__ isle_record_positions,
   const TFloat * __restrict__ prng_vector,
   TFloat * __restrict__ velocities)
  {
    const uint32_t i = blockIdx.x; // ISLE
    const uint32_t j = threadIdx.x; // AGENT

    const uint32_t ISLES = gridDim.x;
    const uint32_t AGENTS = blockDim.x;

    const uint32_t prng_offset = ISLES * AGENTS * DIMENSIONS;
    const uint32_t locus_offset = i * AGENTS + j;

    TFloat * p_g = (TFloat*) &pso_operators_shared_memory[0];

    // Fill common isle records on shared memory
    for(uint32_t k = 0; k < DIMENSIONS; k++) {
      const uint32_t isle_record_positions_idx = k * ISLES + i;
      p_g[k] = isle_record_positions[isle_record_positions_idx];
    }
    __syncthreads();

    // Each thread iterates over a single particle.
    for(uint32_t k = 0; k < DIMENSIONS; k++) {
      const uint32_t particle_gene_idx = k * ISLES * AGENTS + locus_offset;

      const TFloat p_i = record_positions[particle_gene_idx];
      const TFloat x_i = positions[particle_gene_idx];

      const TFloat v_i = velocities[particle_gene_idx];

      const TFloat c_rnd = prng_vector[particle_gene_idx];
      const TFloat s_rnd = prng_vector[particle_gene_idx + prng_offset];

      velocities[particle_gene_idx] =
        inertia_factor * v_i +
        cognitive_factor * c_rnd * (p_i - x_i) +
        social_factor * s_rnd * (p_g[k] - x_i);
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
   const TFloat * record_positions,
   const TFloat * isle_record_positions,
   const TFloat * prng_vector,
   TFloat * velocities)
  {
    canonical_speed_update_kernel
      <<<ISLES, AGENTS, DIMENSIONS * sizeof(TFloat)>>>
      (DIMENSIONS,
       inertia_factor,
       cognitive_factor,
       social_factor,
       positions,
       record_positions,
       isle_record_positions,
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
   const float * record_positions,
   const float * isle_record_positions,
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
   const double * record_positions,
   const double * isle_record_positions,
   const double * prng_vector,
   double * velocities);

  template <typename TFloat>
  __global__
  void canonical_position_update_kernel
  (const uint32_t DIMENSIONS,
   const TFloat * __restrict__ velocities,
   const TFloat * __restrict__ positions,
   TFloat * __restrict__ new_positions)
  {
    const uint32_t i = blockIdx.x; // ISLE
    const uint32_t j = threadIdx.x; // AGENT

    const uint32_t ISLES = gridDim.x;
    const uint32_t AGENTS = blockDim.x;

    const uint32_t locus_offset = i * AGENTS + j;
    // Each thread iterates over a single particle.
    for(uint32_t k = 0; k < DIMENSIONS; k++) {
      const uint32_t particle_gene_idx = locus_offset + k * ISLES * AGENTS;

      const TFloat curr_velocity = velocities[particle_gene_idx];
      const TFloat curr_position = positions[particle_gene_idx];

      new_positions[particle_gene_idx] = curr_position + curr_velocity;
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
