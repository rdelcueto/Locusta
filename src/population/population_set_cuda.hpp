#ifndef LOCUSTA_POPULATION_SET_CUDA_H_
#define LOCUSTA_POPULATION_SET_CUDA_H_

#include <iostream>
#include <limits>

#include "math_constants.h"

#include "population_set.hpp"

namespace locusta {

/**
 * @brief Enum defining the different types of genome copy operations.
 */
enum GenomeCopyKind
{
  GencpyHostToHost,
  GencpyHostToDevice,
  GencpyDeviceToHost,
  GencpyDeviceToDevice
};

/**
 * @brief CUDA implementation of the population_set class.
 *
 * This class extends the population_set class with CUDA-specific functionality.
 *
 * @tparam TFloat Floating-point type.
 */
template<typename TFloat>
struct population_set_cuda : population_set<TFloat>
{

  /**
   * @brief Construct a new population_set_cuda object.
   *
   * @param ISLES Number of isles in the population.
   * @param AGENTS Number of agents per isle.
   * @param DIMENSIONS Number of dimensions per agent.
   */
  population_set_cuda(const uint32_t ISLES,
                      const uint32_t AGENTS,
                      const uint32_t DIMENSIONS);

  /**
   * @brief Destroy the population_set_cuda object.
   */
  virtual ~population_set_cuda();

  /**
   * @brief Swap the data sets.
   *
   * This method swaps the pointers between the current data set and the
   * transformed data set.
   */
  virtual void swap_data_sets();

  /**
   * @brief Copy genomes between populations.
   *
   * This method copies genomes between different populations, supporting
   * various copy kinds (host-to-host, host-to-device, device-to-host,
   * device-to-device).
   *
   * @param dst_data Destination data array.
   * @param src_data Source data array.
   * @param elements Number of elements to copy.
   * @param copy_kind Type of copy operation to perform.
   */
  virtual void gen_cpy(TFloat* dst_data,
                       const TFloat* src_data,
                       size_t elements,
                       GenomeCopyKind copy_kind);

  using population_set<TFloat>::_ISLES;
  using population_set<TFloat>::_AGENTS;
  using population_set<TFloat>::_DIMENSIONS;
  using population_set<TFloat>::_GENES_PER_ISLE;
  using population_set<TFloat>::_TOTAL_AGENTS;
  using population_set<TFloat>::_TOTAL_GENES;
  using population_set<TFloat>::_f_initialized;

  /// CUDA Genomes array.
  TFloat* _dev_data_array;
  /// CUDA Transformed Genomes array.
  TFloat* _dev_transformed_data_array;
  /// CUDA Genome Fitness array.
  TFloat* _dev_fitness_array;
};
}

#include "population_set_cuda_impl.hpp"

#endif /* LOCUSTA_POPULATION_SET_CUDA_H_ */
