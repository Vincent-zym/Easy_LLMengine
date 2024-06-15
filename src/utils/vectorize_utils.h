#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template<typename T>
struct Vec {
    using Type = T;
    static constexpr int size = 0;
};
template<>
struct Vec<half> {
    using Type = half2; 
    static constexpr int size = 2;
};
template<>
struct Vec<float> {
    using Type = float4;
    static constexpr int size = 4;
};