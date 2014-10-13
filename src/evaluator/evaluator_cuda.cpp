namespace locusta {

    template<typename TFloat>
    evaluator_cuda<TFloat>::evaluator_cuda(EvaluationCudaFunctor<TFloat> * eval_functor,
                                           bool f_negate,
                                           BoundMapKind bound_mapping_method,
                                           uint32_t prn_numbers) :
        _evaluation_functor(eval_functor),
        _f_negate(f_negate),
        _bound_mapping_method(bound_mapping_method),
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


    template<typename TFloat>
    __device__
    inline void bound_mapp(BoundMapKind bound_mapping_method,
                           const TFloat &u,
                           const TFloat &l,
                           TFloat &x)
    {
        if (x == x)
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
                    x = CUDART_NAN_F;
                }
                break;
            }
        }
        else
        {
            x = u;
        }
        return;
    }


}
