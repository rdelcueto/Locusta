namespace locusta {

    template<typename TFloat>
    evaluator_cuda<TFloat>::evaluator_cuda(EvaluationCudaFunctor<TFloat> * eval_functor,
                                   bool f_negate,
                                   BoundMapKind bound_mapping,
                                   uint32_t prn_numbers) :
        _evaluation_functor(eval_functor),
        _f_negate(f_negate),
        _bound_mapping(bound_mapping),
        _eval_prn_numbers(prn_numbers)
    {
        // Device Memory Allocation
        CudaSafeCall(cudaMalloc((void **) &_eval_prn_numbers, _eval_prn_size * sizeof(TFloat)));
        CudaCheckError();
    }

    template<typename TFloat>
    evaluator_cuda<TFloat>::~evaluator_cuda()
    {
        CudaSafeCall(cudaFree(_eval_prn_numbers));
    }

    template<typename TFloat>
    void evaluator_cuda<TFloat>::evaluate(evolutionary_solver_cuda * solver)
    {
        (*eval_functor)(solver);
    }
}
