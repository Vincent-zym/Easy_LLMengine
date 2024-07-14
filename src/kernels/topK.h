#pragma once
#include <cuda_runtime.h>
#include <float.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/utils/tensor.h"
template<typename T, int K>
struct topK
{
    T val[K];
    int id[K];

    __device__ void init(){
        for (int i = 0; i < K; i++) {
            id[i] = -1;
            val[i] = -1e-20;
        }
    }

    __device__ void insertHeap(T data, int data_id){
        float v = (float)val[K-1];
        if(id[K-1] == -1 || v < (float)data){
            id[K-1] = data_id;
            val[K-1] = data;
        }
        //Note: 仅需一轮冒泡排序（插入新元素的重排），因为此时除了最后一个新元素，其它都是有序
        for (int i = K - 2; i >= 0; i--){
            if(val[i + 1] > val[i] || id[i] == -1) {
                T tmp = val[i];
                val[i] = val[i + 1];
                val[i + 1] = tmp;                
                int tmp_id = id[i];
                id[i] = id[i + 1];
                id[i + 1] = tmp_id;
            }
        }
    }
};


template <typename T>
void launchTopKforBeamSearch(TensorWrapper<T> *probs,
                             TensorWrapper<int> *topk_ids,
                             TensorWrapper<T> *topk_vals,
                             TensorWrapper<int> *final_topk_ids,
                             TensorWrapper<T> *final_topk_vals);
// template<typename T>
// void launchTopKforBeamSearch(const T* probs, 
//                             const int batch_size,
//                             const int vocab_size, 
//                             int* topk_ids,
//                             T* topk_vals,
//                             int* final_topk_ids,
//                             T* final_topk_vals);
