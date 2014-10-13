#include "evaluator.hpp"
#include <limits>

namespace locusta {



    template<typename TFloat>
    evaluator<TFloat>::evaluator(EvaluationFunctor<TFloat> * eval_functor,
                         bool f_negate,
                         BoundMapKind bound_mapping_method,
                         uint32_t prn_numbers)
        : _evaluation_functor(eval_functor),
          _f_negate(f_negate),
          _bound_mapping_method(bound_mapping_method),
          _eval_prn_size(prn_numbers)
    {
        _eval_prn_numbers = new TFloat[_eval_prn_size];
    }

    template<typename TFloat>
    evaluator<TFloat>::~evaluator()
    {
        delete [] _eval_prn_numbers;
    }

    /// Evaluate the solver's population data set.
    template<typename TFloat>
    void evaluator<TFloat>::evaluate(evolutionary_solver<TFloat> * solver)
    {
        (*_evaluation_functor)(solver);
    }

    template<typename TFloat>
    inline void evaluator<TFloat>::bound_map(BoundMapKind bound_mapping_method,
                                             const TFloat &u,
                                             const TFloat &l,
                                             TFloat &x)
    {
        switch (bound_mapping_method)
        {
        case BoundMapKind::CropBounds: /// Out of bounds Crop
            x = x > u ? u : x < l ? l : x;
            break;
        case BoundMapKind::MirrorBounds: /// Out of bounds Mirror
            x = x > u ? (2*u - x) : x < l ? (2*l - x) : x;
            break;
        case BoundMapKind::IgnoreBounds: /// Out of bounds Error
            if ( x > u || x < l )
            {
                x = std::numeric_limits<TFloat>::quiet_NaN();
                std::cerr << "Out of bounds gene!" << std::endl;
            }
            break;
        }
        return;
    }

} // namespace locusta
