#ifndef LOCUSTA_POPULATION_SET_CUDA_H_
#define LOCUSTA_POPULATION_SET_CUDA_H_

#include <iostream>
#include <limits>

#include "math_constants.h"

#include "population_set.hpp"

namespace locusta {
    enum GenomeCopyKind { GencpyHostToHost, GencpyHostToDevice, GencpyDeviceToHost, GencpyDeviceToDevice };

    template <typename TFloat>
    struct population_set_cuda : population_set<TFloat> {

        population_set_cuda(const uint32_t ISLES,
                            const uint32_t AGENTS,
                            const uint32_t DIMENSIONS);

        virtual ~population_set_cuda();

        /// Wrapper to call swap_dev_data_sets.
        virtual void swap_data_sets();

        /// Copies genomes between populations.
        virtual void gen_cpy(TFloat * dst_data,
                             const TFloat * src_data,
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
        TFloat * _dev_data_array;
        /// CUDA Transformed Genomes array.
        TFloat * _dev_transformed_data_array;
        /// CUDA Genome Fitness array.
        TFloat * _dev_fitness_array;

    };
}

#include "population_set_cuda_impl.hpp"

#endif /* LOCUSTA_POPULATION_SET_CUDA_H_ */
