#ifndef LOCUSTA_PRECISION_TYPES_HPP_
#define LOCUSTA_PRECISION_TYPES_HPP_

/**
 * @file precision_types.hpp
 * @brief Declarations, stream operators, and arithmetic overloads for low-precision floating-point types.
 *
 * Provides support for NVIDIA low-precision types (__half, __nv_bfloat16, __nv_fp8_e4m3, __nv_fp8_e5m2)
 * across both host (CPU) and device (CUDA) contexts.
 *
 * Rationale:
 * - Centralizes NVIDIA low-precision type definitions and compatibility shims in a single, reusable header.
 * - Provides missing operator overloads (arithmetic, relational, and stream insertion) for FP8 types that
 *   are absent from standard CUDA Toolkit headers.
 * - Decouples low-precision type support from population_set.hpp and other container components.
 */

#include <iostream>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

inline std::ostream& operator<<(std::ostream& os, const __half& v) {
  return os << static_cast<float>(v);
}
inline std::ostream& operator<<(std::ostream& os, const __nv_bfloat16& v) {
  return os << static_cast<float>(v);
}
inline std::ostream& operator<<(std::ostream& os, const __nv_fp8_e4m3& v) {
  return os << static_cast<float>(v);
}
inline std::ostream& operator<<(std::ostream& os, const __nv_fp8_e5m2& v) {
  return os << static_cast<float>(v);
}

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

#define DEFINE_FP8_OPS(T) \
__host__ __device__ inline bool operator>(const T& a, const T& b) { return static_cast<float>(a) > static_cast<float>(b); } \
__host__ __device__ inline bool operator<(const T& a, const T& b) { return static_cast<float>(a) < static_cast<float>(b); } \
__host__ __device__ inline bool operator>=(const T& a, const T& b) { return static_cast<float>(a) >= static_cast<float>(b); } \
__host__ __device__ inline bool operator<=(const T& a, const T& b) { return static_cast<float>(a) <= static_cast<float>(b); } \
__host__ __device__ inline bool operator==(const T& a, const T& b) { return static_cast<float>(a) == static_cast<float>(b); } \
__host__ __device__ inline bool operator!=(const T& a, const T& b) { return static_cast<float>(a) != static_cast<float>(b); } \
__host__ __device__ inline T operator+(const T& a, const T& b) { return T(static_cast<float>(a) + static_cast<float>(b)); } \
__host__ __device__ inline T operator-(const T& a, const T& b) { return T(static_cast<float>(a) - static_cast<float>(b)); } \
__host__ __device__ inline T operator*(const T& a, const T& b) { return T(static_cast<float>(a) * static_cast<float>(b)); } \
__host__ __device__ inline T operator/(const T& a, const T& b) { return T(static_cast<float>(a) / static_cast<float>(b)); } \
__host__ __device__ inline T operator-(const T& a) { return T(-static_cast<float>(a)); } \
__host__ __device__ inline T& operator+=(T& a, const T& b) { a = T(static_cast<float>(a) + static_cast<float>(b)); return a; } \
__host__ __device__ inline T& operator-=(T& a, const T& b) { a = T(static_cast<float>(a) - static_cast<float>(b)); return a; } \
__host__ __device__ inline T& operator*=(T& a, const T& b) { a = T(static_cast<float>(a) * static_cast<float>(b)); return a; } \
__host__ __device__ inline T& operator/=(T& a, const T& b) { a = T(static_cast<float>(a) / static_cast<float>(b)); return a; }

DEFINE_FP8_OPS(__nv_fp8_e4m3)
DEFINE_FP8_OPS(__nv_fp8_e5m2)

#endif // LOCUSTA_PRECISION_TYPES_HPP_
