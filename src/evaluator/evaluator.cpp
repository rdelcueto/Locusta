#include "evaluator.hpp"

namespace locusta {

    template<typename TFloat>
    evaluator<TFloat>::evaluator(EvaluationFunctor<TFloat> * eval_functor,
                         bool f_negate,
                         BoundMapKind bound_mapping,
                         uint32_t prn_numbers)
        : _evaluation_functor(eval_functor),
          _f_negate(f_negate),
          _bound_mapping(bound_mapping),
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
} // namespace locusta
