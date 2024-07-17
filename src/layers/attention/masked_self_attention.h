#pragma once
#include "src/weights/llama/attention_weights.h"
#include "src/memory/allocator/cuda_allocator.h"
#include "src/kernels/linear.h" //1st/4th kernel of masked self attention, qkv gemm
#include "src/kernels/attn_softmax_kernel.h"
#include "src/kernels/qkv_bias_and_RoPE.h" //2nd rope
#include "src/kernels/fused_decoder_self_attention.h" //3rd kernel 
#include "src/utils/tensor.h"
#include "src/kernels/cublas_utils.h"
#include "src/models/llama/llama_params.h"
#include "src/utils/macro.h"

// (RussWong)note: 这里面的数据成员都是只存在于attention layer，而不像finished，seq lengths这种贯穿整个过程
template<typename T>
class LLaMASelfAttentionLayer {
private:
    // this params are shared across all LLMs
    const int head_num;
    const int head_size;
    const int hidden_units;
    // for GQA and MQA，多少组q共享一组kv
    // 注意llama2-7b attn部分是MHA, 13b及其以上是GQA
    const int q_head_per_kv;
    const int kv_head_num;
    float scale;
    // this params are only saw in llama and are unchanged 
    LLaMAAttentionStaticParams attn_static_params;
    cudaStream_t stream;
    BaseAllocator* allocator;
    // for linear and batchgemm
    cublasWrapper* cublas_wrapper;
    // forward推理所需的中间buffer
    TensorWrapper<T>* qkv_buf     = nullptr; // for qkv linear output and rope input/output
    TensorWrapper<T>* mha_output = nullptr; // mha output, then invoke a linear to attention output

public:
    LLaMASelfAttentionLayer(int head_num,
                               int kv_head_num,
                               int head_size,
                               LLaMAAttentionStaticParams attn_params,
                               cudaStream_t stream,
                               cublasWrapper* cublas_wrapper,
                               BaseAllocator* allocator);
    // private数据成员只能通过成员函数来访问
    LLaMAAttentionStaticParams& GetAttnStaticParams(){
        return attn_static_params;
    }
    // 分配forward所需buffer
    void allocForForward(LLaMAAttentionDynParams& params);
    // forward结束后释放所占用buffer
    void freeBuf();
    // forward，即推理
    void forward(TensorMap& inputs, TensorMap& outputs, LLaMAattentionWeights<T>& weights, LLaMAAttentionDynParams& params);
};
