#include "src/kernels/build_casual_mask.h"
// mask shape =  [bs, max_q_len, max_k_len]
template<typename T>
__global__ void BuildCausalMasksConsideringContextPastKV(T* mask,
                                                const int* q_lens,  //input lens, shape=[batch size]
                                                const int* k_lens,  //context lens, shape=[batch size]
                                                int max_q_len,  // max(q_lens)
                                                int max_k_len){ // max(k_lens)
    int tid = threadIdx.x;
    int qlen = q_lens[blockIdx.x];
    int klen = k_lens[blockIdx.x];
    mask += blockIdx.x * max_q_len * max_k_len;
    int offset = threadIdx.x;
    // note: this judgement confirms we dont exceed data boundry
    while (offset < max_q_len * max_k_len){
        // 分别求出行号q和列号k
        int q = offset / max_k_len;
        int k = offset % max_k_len;
        // 此处与视频中的代码不同，k考虑了多轮对话的上下文序列，但设置mask时 k >= klen - qlen 将旧序列一并遮去了
        // 下图为支持多轮对话的mask
        // 1 1 1 | 1 -inf -inf
        // 1 1 1 | 1   1  -inf
        // 1 1 1 | 1   1    1
        // 下图为不支持多轮对话的mask
        // -inf -inf -inf | 1 -inf -inf
        // -inf -inf -inf | 1   1  -inf
        // -inf -inf -inf | 1   1    1
        // "|"符号前表示旧的对话序列，符号后表示当前轮的对话序列
        bool is_one = q < qlen && k < klen && k <= q + (klen - qlen) && k >= klen - qlen;
        mask[offset] = static_cast<T>(is_one);

        offset += blockDim.x;
    }
}

template<typename T>
void launchBuildCausalMasks(TensorWrapper<T>* mask, 
                            TensorWrapper<int>* q_lens, 
                            TensorWrapper<int>* k_lens)
{
    int batch_size = mask->shape[0];
    int max_q_len = mask->shape[1];
    int max_k_len = mask->shape[2];
    BuildCausalMasksConsideringContextPastKV<T><<<batch_size, 256>>>(mask->data, q_lens->data, k_lens->data, max_q_len, max_k_len);
}

template void launchBuildCausalMasks(TensorWrapper<float>* mask, 
                            TensorWrapper<int>* q_lens, 
                            TensorWrapper<int>* k_lens);

template void launchBuildCausalMasks(TensorWrapper<half>* mask, 
                            TensorWrapper<int>* q_lens, 
                            TensorWrapper<int>* k_lens);
