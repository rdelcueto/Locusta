#ifndef _BENCHMARKS_CPU_IMPL_H_
#define _BENCHMARKS_CPU_IMPL_H_

namespace locusta {

    template<typename TFloat>
    inline TFloat scale(const TFloat x, const TFloat scale) {
        return x * scale;
    }

    template<typename TFloat>
    inline TFloat shift(const TFloat x, const TFloat o) {
        return x - o;
    }

    template<typename TFloat>
    inline TFloat asy(const TFloat x, const TFloat beta, const uint32_t i, const uint32_t k) {
        if(x > 0) {
            return pow(x, 1 + beta*i/(k-1)*pow(x, 0.5));
        } else {
            return x;
        }
    }

    template<typename TFloat>
    inline TFloat osz(const TFloat x, const uint32_t i, const uint32_t k) {
        const TFloat c3 = 0.049;

        if (i == 0 || i == (k-1)) {

            if (x == 0) {
                return 0;
            } else {
                const TFloat xx = log(fabs(x));
                if (x > 0) {
                    const TFloat c1 = 10;
                    const TFloat c2 = 7.9;

                    return exp(xx + c3*(sin(c1*xx) + sin(c2*xx)));
                } else {
                    const TFloat c1 = 5.5;
                    const TFloat c2 = 3.1;

                    return -exp(xx + c3*(sin(c1*xx) + sin(c2*xx)));
                }
            }

        } else {
            return x;
        }
    }

    template<typename TFloat>
    inline TFloat sphere(const uint32_t DIMENSIONS,
                         const uint32_t DIMENSION_OFFSET,
                         const TFloat * evaluation_vector) {
        TFloat reduction_sum = 0.0;

#pragma omp simd reduction(+:reduction_sum) linear(evaluation_vector:DIMENSION_OFFSET*sizeof(TFloat))
        for (uint32_t k = 0; k < DIMENSIONS; ++k) {
            const TFloat x = evaluation_vector[k * DIMENSION_OFFSET];
            reduction_sum += x * x;
        }
        return reduction_sum;
    }

    template<typename TFloat>
    inline TFloat ellips(const uint32_t DIMENSIONS,
                         const uint32_t DIMENSION_OFFSET,
                         const TFloat * evaluation_vector) {
        const TFloat c1 = 10.0;
        const TFloat c2 = 6.0;

        TFloat reduction_sum = 0.0;

#pragma omp simd reduction(+:reduction_sum) linear(evaluation_vector:DIMENSION_OFFSET*sizeof(TFloat))
        for (uint32_t k = 0; k < DIMENSIONS; ++k) {
            const TFloat x = evaluation_vector[k * DIMENSION_OFFSET];
            const TFloat osz_x = osz<TFloat>(x, k, DIMENSIONS);
            reduction_sum += pow(c1, c2 * k/(DIMENSIONS-1))*osz_x*osz_x;
        }
        return reduction_sum;
    }

    template<typename TFloat>
    inline TFloat bent_cigar(const uint32_t DIMENSIONS,
                             const uint32_t DIMENSION_OFFSET,
                             const TFloat * evaluation_vector) {
        const TFloat beta = 0.5;
        const TFloat c1 = 10.0;
        const TFloat c2 = 6.0;
        TFloat reduction_sum = 0;

#pragma omp simd reduction(+:reduction_sum) linear(evaluation_vector:DIMENSION_OFFSET*sizeof(TFloat))
        for (uint32_t k = 0; k < DIMENSIONS; ++k) {
            const TFloat x = evaluation_vector[k * DIMENSION_OFFSET];
            const TFloat asy_x = asy<TFloat>(x, beta, k, DIMENSIONS);
            if (k == 0) {
                reduction_sum += asy_x * asy_x;
            } else {
                reduction_sum += pow(c1, c2) * asy_x * asy_x;
            }
        }

        return reduction_sum;
    }

    template<typename TFloat>
    inline TFloat discus(const uint32_t DIMENSIONS,
                         const uint32_t DIMENSION_OFFSET,
                         const TFloat * evaluation_vector) {
        const TFloat c1 = 10.0;
        const TFloat c2 = 6.0;

        TFloat reduction_sum = 0.0;

#pragma omp simd reduction(+:reduction_sum) linear(evaluation_vector:DIMENSION_OFFSET*sizeof(TFloat))
        for (uint32_t k = 0; k < DIMENSIONS; ++k) {
            const TFloat x = evaluation_vector[k * DIMENSION_OFFSET];
            const TFloat osz_x = osz<TFloat>(x, k, DIMENSIONS);
            if (k == 0) {
                reduction_sum += pow(c1, c2) * osz_x * osz_x;
            } else {
                reduction_sum += osz_x * osz_x;
            }
        }
        return reduction_sum;
    }

    template<typename TFloat>
    inline TFloat diff_powers(const uint32_t DIMENSIONS,
                              const uint32_t DIMENSION_OFFSET,
                              const TFloat * evaluation_vector) {
        const TFloat c1 = 0.5;
        TFloat reduction_sum = 0.0;

#pragma omp simd reduction(+:reduction_sum) linear(evaluation_vector:DIMENSION_OFFSET*sizeof(TFloat))
        for (uint32_t k = 0; k < DIMENSIONS; ++k) {
            const TFloat x = evaluation_vector[k * DIMENSION_OFFSET];
            reduction_sum += pow(fabs(x), 2+4*k/(DIMENSIONS-1));
        }

        reduction_sum = pow(reduction_sum, c1);
        return reduction_sum;
    }

    template<typename TFloat>
    inline TFloat rosenbrock(const uint32_t DIMENSIONS,
                             const uint32_t DIMENSION_OFFSET,
                             const TFloat * evaluation_vector) {
        TFloat reduction_sum = 0.0;
        TFloat x = evaluation_vector[0] + 1;

        for (uint32_t k = 0; k < DIMENSIONS-1; ++k) {
            const TFloat x_1 = evaluation_vector[(k+1) * DIMENSION_OFFSET] + 1;
            const TFloat y = x * x - x_1;
            const TFloat z = x - 1.0;
            reduction_sum += 100 * y * y + z * z;

            x = x_1; // Update x value.
        }
        return reduction_sum;
    }

    template<typename TFloat>
    inline TFloat schaffer(const uint32_t DIMENSIONS,
                           const uint32_t DIMENSION_OFFSET,
                           const TFloat * evaluation_vector) {
        const TFloat beta = 0.5;
        const TFloat c1 = 10.0;
        const TFloat c2 = 1.0;
        const TFloat c3 = 2.0;
        const TFloat c4 = 0.5;
        const TFloat c5 = 0.2;

        const TFloat x = evaluation_vector[0];
        const TFloat asy_x = asy<TFloat>(x, beta, 0, DIMENSIONS);
        TFloat y = asy_x * pow(c1, c2*0/(DIMENSIONS-1)/c3);

        TFloat reduction_sum = 0.0;
        for (uint32_t k = 0; k < DIMENSIONS-1; ++k) {
            const TFloat x_1 = evaluation_vector[(k+1) * DIMENSION_OFFSET];
            const TFloat asy_x_1 = asy<TFloat>(x_1, beta, (k+1), DIMENSIONS);
            const TFloat y_1 = asy_x * pow(c1, c2*k/(DIMENSIONS-1)/c3);
            const TFloat z = pow(y*y + y_1*y_1, c4);

            const TFloat tmp = sin(50*pow(z, c5));
            reduction_sum += pow(z, 0.5) + pow(z, 0.5) * tmp*tmp;

            y = y_1;
        }
        return reduction_sum;
    }

    template<typename TFloat>
    inline TFloat ackley(const uint32_t DIMENSIONS,
                         const uint32_t DIMENSION_OFFSET,
                         const TFloat * evaluation_vector) {
        const TFloat PI = 3.1415926535;
        const TFloat E = 2.7182818284;
        const TFloat beta = 0.5;
        const TFloat c1 = 10.0;
        const TFloat c2 = 1.0;
        const TFloat c3 = 2.0;
        const TFloat c4 = 0.2;
        const TFloat c5 = 20.0;

        TFloat reduction_sum_a = 0.0;
        TFloat reduction_sum_b = 0.0;

#pragma omp simd reduction(+:reduction_sum_a, reduction_sum_b)
        for (uint32_t k = 0; k < DIMENSIONS; ++k) {
            const TFloat x = evaluation_vector[k * DIMENSION_OFFSET];
            const TFloat asy_x = asy<TFloat>(x, beta, k, DIMENSIONS);
            const TFloat y = asy_x * pow(c1, c2*k/(DIMENSIONS-1)/c3);

            reduction_sum_a += y * y;
            reduction_sum_b += cos(c3*PI*y);
        }

        reduction_sum_a = -c4*sqrt(reduction_sum_a/DIMENSIONS);
        reduction_sum_b /= DIMENSIONS;

        return E - c5 * exp(reduction_sum_a) - exp(reduction_sum_b) + c5;
    }

    template<typename TFloat>
    inline TFloat weierstrass(const uint32_t DIMENSIONS,
                              const uint32_t DIMENSION_OFFSET,
                              const TFloat * evaluation_vector) {
        const TFloat PI = 3.14159;
        const TFloat beta = 0.5;
        const TFloat c1 = 10.0;
        const TFloat c2 = 1.0;
        const TFloat c3 = 2.0;
        const TFloat c4 = 0.5;

        const TFloat a = 0.5;
        const TFloat b = 3.0;
        const uint32_t it_max = 20;

        TFloat reduction_sum_a = 0.0;
        TFloat reduction_sum_b;
        TFloat reduction_sum_c;

        for (uint32_t k = 0; k < DIMENSIONS; ++k) {
            reduction_sum_b = 0.0;
            reduction_sum_c = 0.0;

            const TFloat x = evaluation_vector[k * DIMENSION_OFFSET];
            const TFloat asy_x = asy<TFloat>(x, beta, k, DIMENSIONS);
            const TFloat y = asy_x * pow(c1, c2*k/(DIMENSIONS-1)/c3);

#pragma omp simd reduction(+:reduction_sum_b, reduction_sum_c)
            for (uint32_t it = 0; it <= it_max; ++it) {
                reduction_sum_b += pow(a, it) * cos(c3*PI*pow(b, it)*(y + c4));
                reduction_sum_c += pow(a, it) * cos(c3*PI*pow(b, it)*c4);
            }

            reduction_sum_a += reduction_sum_b;
        }
        return reduction_sum_a - DIMENSIONS * reduction_sum_c;
    }

    template<typename TFloat>
    inline TFloat griewank(const uint32_t DIMENSIONS,
                           const uint32_t DIMENSION_OFFSET,
                           const TFloat * evaluation_vector) {
        const TFloat c1 = 100.0;
        const TFloat c2 = 1.0;
        const TFloat c3 = 2.0;
        const TFloat c4 = 1.0;
        const TFloat c5 = 4000.0;

        TFloat reduction_sum = 0.0;
        TFloat reduction_prod = 1.0;

#pragma omp simd reduction(+:reduction_sum) reduction(*:reduction_prod)
        for (uint32_t k = 0; k < DIMENSIONS; ++k) {
            const TFloat x = evaluation_vector[k * DIMENSION_OFFSET];
            const TFloat y = x*pow(c1, c2*k/(DIMENSIONS-1)/c3);
            reduction_sum += y*y;
            reduction_prod *= cos(y/sqrt(c2+k));
        }
        return c4 + reduction_sum/c5 - reduction_prod;
    }

    template<typename TFloat>
    inline TFloat rastrigin(const uint32_t DIMENSIONS,
                            const uint32_t DIMENSION_OFFSET,
                            const TFloat * evaluation_vector) {

        const TFloat PI = 3.14159;
        const TFloat alpha = 10.0;
        const TFloat beta = 0.2;
        const TFloat c1 = 1.0;
        const TFloat c2 = 2.0;
        const TFloat c3 = 10.0;

        TFloat reduction_sum = 0.0;

#pragma omp simd reduction(+:reduction_sum) linear(evaluation_vector:DIMENSION_OFFSET*sizeof(TFloat))
        for (uint32_t k = 0; k < DIMENSIONS; ++k) {
            const TFloat x = evaluation_vector[k * DIMENSION_OFFSET];
            const TFloat osz_x = osz<TFloat>(x, k, DIMENSIONS);
            const TFloat asy_x = asy<TFloat>(osz_x, beta, k, DIMENSIONS);
            const TFloat y = pow(alpha, c1*k/(DIMENSIONS-1)/2);

            reduction_sum += (y*y - c3*cos(c2*PI*y)) + c3;
        }

        return reduction_sum;
    }

    template<typename TFloat>
    void benchmark_dispatch
    (const uint32_t ISLES,
     const uint32_t AGENTS,
     const uint32_t DIMENSIONS,
     const bool F_NEGATE_EVALUATION,
     const uint32_t FUNC_ID,
     const TFloat EVALUATION_BIAS,
     const TFloat * SHIFT_ORIGIN,
     const bool F_ROTATE,
     const TFloat * ROTATION_MATRIX,
     const TFloat * evaluation_data,
     TFloat * evaluation_results,
     prngenerator_cpu<TFloat> * local_generator) {

        const uint32_t REPETITIONS = 1e2;

#pragma omp parallel for collapse(2)
        for(uint32_t i = 0; i < ISLES; ++i) {
            for(uint32_t j = 0; j < AGENTS; ++j) {
                const uint32_t isle = i;
                const uint32_t agent = j;
                const uint32_t OFFSET = 1;

                TFloat result = 0;
                const TFloat * genome = evaluation_data + i * AGENTS * DIMENSIONS + j * DIMENSIONS;
                for(uint32_t r = 0; r < REPETITIONS; ++r) {
                    switch (FUNC_ID) {

                    case BenchmarkFunctor<TFloat>::FUNCTION_IDENTIFIERS::ROT_ELLIPS:
                        result = ellips(DIMENSIONS,
                                        OFFSET,
                                        genome);
                        break;
                    case BenchmarkFunctor<TFloat>::FUNCTION_IDENTIFIERS::ROT_BENT_CIGAR:
                        result = bent_cigar(DIMENSIONS,
                                            OFFSET,
                                            genome);
                        break;
                    case BenchmarkFunctor<TFloat>::FUNCTION_IDENTIFIERS::ROT_DISCUS:
                        result = discus(DIMENSIONS,
                                        OFFSET,
                                        genome);
                        break;
                    case BenchmarkFunctor<TFloat>::FUNCTION_IDENTIFIERS::DIFF_POWERS:
                        result = diff_powers(DIMENSIONS,
                                             OFFSET,
                                             genome);
                        break;
                    case BenchmarkFunctor<TFloat>::FUNCTION_IDENTIFIERS::ROT_ROSENBROCK:
                        result = rosenbrock(DIMENSIONS,
                                            OFFSET,
                                            genome);
                        break;
                    case BenchmarkFunctor<TFloat>::FUNCTION_IDENTIFIERS::ROT_SCHAFFER:
                        result = schaffer(DIMENSIONS,
                                          OFFSET,
                                          genome);
                        break;
                    case BenchmarkFunctor<TFloat>::FUNCTION_IDENTIFIERS::ROT_ACKLEY:
                        result = ackley(DIMENSIONS,
                                        OFFSET,
                                        genome);
                        break;
                    case BenchmarkFunctor<TFloat>::FUNCTION_IDENTIFIERS::ROT_WEIERSTRASS:
                        result = weierstrass(DIMENSIONS,
                                             OFFSET,
                                             genome);
                        break;
                    case BenchmarkFunctor<TFloat>::FUNCTION_IDENTIFIERS::ROT_GRIEWANK:
                        result = griewank(DIMENSIONS,
                                          OFFSET,
                                          genome);
                        break;
                    case BenchmarkFunctor<TFloat>::FUNCTION_IDENTIFIERS::RASTRIGIN:
                        result = rastrigin(DIMENSIONS,
                                           OFFSET,
                                           genome);
                        break;
                    case BenchmarkFunctor<TFloat>::FUNCTION_IDENTIFIERS::ROT_RASTRIGIN:
                        result = rastrigin(DIMENSIONS,
                                           OFFSET,
                                           genome);
                        break;

                    case BenchmarkFunctor<TFloat>::FUNCTION_IDENTIFIERS::SPHERE:
                    default:
                        result = sphere(DIMENSIONS,
                                        OFFSET,
                                        genome);
                        break;
                    }
                }

                evaluation_results[isle * AGENTS + agent] = F_NEGATE_EVALUATION ?
                    -result : result;
            }
        }
    }

}

#endif /* _BENCHMARKS_CPU_IMPL */
