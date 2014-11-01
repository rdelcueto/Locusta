#include <limits>

#include "evaluator_cpu.hpp"

namespace locusta {

    template<typename TFloat>
    evaluator_cpu<TFloat>::evaluator_cpu(EvaluationFunctor<TFloat> * eval_functor,
                                         bool f_negate,
                                         BoundMapKind bound_mapping_method,
                                         uint32_t prn_numbers)
        : evaluator<TFloat>(eval_functor,
                            f_negate,
                            bound_mapping_method,
                            prn_numbers)
    {
        _eval_prn_numbers = new TFloat[_eval_prn_size];
    }

    template<typename TFloat>
    evaluator_cpu<TFloat>::~evaluator_cpu()
    {
        delete [] _eval_prn_numbers;
    }

    /// Evaluate the solver's population data set.
    template<typename TFloat>
    void evaluator_cpu<TFloat>::evaluate(evolutionary_solver<TFloat> * solver)
    {
        (*_evaluation_functor)(dynamic_cast<evolutionary_solver_cpu<TFloat> *>(solver));
    }

    template<typename TFloat>
    inline void evaluator_cpu<TFloat>::bound_map(BoundMapKind bound_mapping_method,
                                                 const TFloat &u,
                                                 const TFloat &l,
                                                 TFloat &x)
    {
        switch (bound_mapping_method)
        {
        case CropBounds: /// Out of bounds Crop
            x = x > u ? u : x < l ? l : x;
            break;
        case MirrorBounds: /// Out of bounds Mirror
            x = x > u ? (2*u - x) : x < l ? (2*l - x) : x;
            break;
        case IgnoreBounds: /// Out of bounds Error
            break;
        }
        return;
    }

} // namespace locusta
