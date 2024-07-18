#include <random>
#include "src/weights/llama/layer_weights.h"
#include "src/utils/macro.h"
template<typename T>
LlamaLayerWeight<T>::LlamaLayerWeight(int     head_num,
                                    int     kv_head_num,
                                    int     head_size,
                                    int     inter_size,
                                    WeightType weight_type,
                                    bool       attn_bias):
    head_num(head_num),
    kv_head_num(kv_head_num),
    head_size(head_size),
    hidden_units(head_num * head_size),
    inter_size(inter_size),
    weight_type(weight_type),
    attn_bias(attn_bias)
{
    // init weights structure and cudamalloc for weights
    CHECK(cudaMalloc((void**)&attn_norm_weight.gamma, sizeof(T) * hidden_units));
    CHECK(cudaMalloc((void**)&ffn_norm_weight.gamma, sizeof(T) * hidden_units));
    self_attn_weight.qkv.type = weight_type;
    self_attn_weight.qkv.shape = {(head_num + 2 * kv_head_num) * head_size, hidden_units};
    CHECK(cudaMalloc((void**)&self_attn_weight.qkv.data, sizeof(T) * hidden_units * (head_num + 2 * kv_head_num) * head_size));
    self_attn_weight.output.type = weight_type;
    self_attn_weight.output.shape = {hidden_units, hidden_units};
    CHECK(cudaMalloc((void**)&self_attn_weight.output.data, sizeof(T) * hidden_units * hidden_units));
    if (attn_bias) {
        CHECK(cudaMalloc((void**)&self_attn_weight.qkv.bias, sizeof(T) * (head_num + 2 * kv_head_num) * head_size));
        CHECK(cudaMalloc((void**)&self_attn_weight.output.bias, sizeof(T) * hidden_units));
    }
    // (RussWong)note: we concat gate linear weight and up linear weight to one weight tensor for performance improvement
    ffn_weight.gateAndup.type = weight_type;
    ffn_weight.down.type = weight_type;
    ffn_weight.gateAndup.shape = {2 * inter_size, hidden_units};
    // ffn_weight.up.shape = {hidden_units, inter_size};
    ffn_weight.down.shape = {hidden_units, inter_size};
    CHECK(cudaMalloc((void**)&ffn_weight.gateAndup.data, sizeof(T) * hidden_units * 2 * inter_size));
    // CHECK(cudaMalloc((void**)&ffn_weight.up.data, hidden_units * inter_size));
    CHECK(cudaMalloc((void**)&ffn_weight.down.data, sizeof(T) * hidden_units * inter_size));
}
// (RussWong)note: weight from HF is always half type, and if we want run fp32 inference, we should convert half weight to fp32 weight in tools/weights_convert.py 
// (RussWong)note: shape and data of ffn weight downloaded form HF are transposed, so we should carefully declare shape here
template<typename T>
void LlamaLayerWeight<T>::loadWeights(std::string weight_path, WeightType weight_type) // weighttype参数比较多余
{
    loadWeightFromBin<T, float>::internalFunc(attn_norm_weight.gamma, {hidden_units}, weight_path + ".input_layernorm.weight.bin");
    loadWeightFromBin<T, float>::internalFunc(ffn_norm_weight.gamma, {hidden_units}, weight_path + ".post_attention_layernorm.weight.bin");

    loadWeightFromBin<T, float>::internalFunc(self_attn_weight.qkv.data, {(head_num + 2 * kv_head_num) * head_size, hidden_units}, weight_path + ".self_attn.qkv.weight.bin");
    loadWeightFromBin<T, float>::internalFunc(self_attn_weight.output.data, {hidden_units, hidden_units}, weight_path + ".self_attn.o_proj.weight.bin");
    loadWeightFromBin<T, float>::internalFunc(ffn_weight.gateAndup.data, {2 * inter_size, hidden_units}, weight_path + ".mlp.gate_up_proj.weight.bin");
    // loadWeightFromBin<T, float>::internalFunc(ffn_weight.up.data, {hidden_units, inter_size}, weight_path + ".mlp.up_proj.weight.bin");
    loadWeightFromBin<T, float>::internalFunc(ffn_weight.down.data, {hidden_units, inter_size}, weight_path + ".mlp.down_proj.weight.bin");
    if (attn_bias) {//TODO
        loadWeightFromBin<T, float>::internalFunc(self_attn_weight.qkv.bias, {(head_num + 2 * kv_head_num) * head_size}, weight_path + ".attention.wqkv.bias.bin");
        loadWeightFromBin<T, float>::internalFunc(self_attn_weight.output.bias, {head_num *  head_size}, weight_path + ".attention.wo.bias.bin");
    } else {
    	self_attn_weight.qkv.bias = nullptr;
	self_attn_weight.output.bias = nullptr;
	ffn_weight.down.bias = nullptr;
    } 
    // (RussWong)note: below code lines can be enabled when I dont support qkvbiasandrope and fusedbiasaddresidual's bias nullptr case.
    //T* d_dummy_qkv_bias;
    //GPUMalloc(&d_dummy_qkv_bias, sizeof(T) * (head_num + 2 * kv_head_num) * head_size);
    //cudaMemset((void*)d_dummy_qkv_bias, 0, sizeof(T) * (head_num + 2 * kv_head_num) * head_size);
    //self_attn_weight.qkv.bias = (T*)d_dummy_qkv_bias;

    //T* d_dummy_output_bias;
    //GPUMalloc(&d_dummy_output_bias, sizeof(T) * head_num *  head_size);
    //cudaMemset((void*)d_dummy_output_bias, 0, sizeof(T) * head_num *  head_size);
    //self_attn_weight.output.bias = (T*)d_dummy_output_bias;

    //T* d_dummy_ffn_down_bias;
    //GPUMalloc(&d_dummy_ffn_down_bias, sizeof(T) * hidden_units);
    //cudaMemset((void*)d_dummy_ffn_down_bias, 0, sizeof(T) * hidden_units);
    //ffn_weight.down.bias = (T*)d_dummy_ffn_down_bias;
}

// (RussWong)note: load dummy model/weight API, is used to the time when you want test inference performance only
template<typename T>
void LlamaLayerWeight<T>::loadWeights() 
{
    T* d_dummy_attn_norm_weight;
    T* d_dummy_ffn_norm_weight;
    T* d_dummy_qkv_weights;
    //T* d_dummy_qkv_bias;
    T* d_dummy_output_weights;
    T* d_dummy_output_bias;
    T* d_dummy_ffn_down;
    T* d_dummy_ffn_down_bias;
    T* d_dummy_ffn_gate_up;
    // T* d_dummy_ffn_up;
    CHECK(cudaMalloc((void**)&d_dummy_attn_norm_weight, sizeof(T) * hidden_units));
    CHECK(cudaMalloc((void**)&d_dummy_ffn_norm_weight, sizeof(T) * hidden_units));
    CHECK(cudaMalloc((void**)&d_dummy_qkv_weights, sizeof(T) * hidden_units * (head_num + 2 * kv_head_num) * head_size));
   // CHECK(cudaMalloc((void**)&d_dummy_qkv_bias, sizeof(T) * (head_num + 2 * kv_head_num) * head_size));
    CHECK(cudaMalloc((void**)&d_dummy_output_weights, sizeof(T) * hidden_units * hidden_units));
    CHECK(cudaMalloc((void**)&d_dummy_output_bias, sizeof(T) * hidden_units));
    CHECK(cudaMalloc((void**)&d_dummy_ffn_down, sizeof(T) * hidden_units * inter_size));
    CHECK(cudaMalloc((void**)&d_dummy_ffn_down_bias, sizeof(T) * hidden_units));
    CHECK(cudaMalloc((void**)&d_dummy_ffn_gate_up, sizeof(T) * hidden_units * 2 * inter_size));
    // CHECK(cudaMalloc(&d_dummy_ffn_up, sizeof(T) * hidden_units * inter_size));

    T* h_dummy_attn_norm_weight = (T*)malloc(sizeof(T) * hidden_units);
    T* h_dummy_ffn_norm_weight = (T*)malloc(sizeof(T) * hidden_units);
    T* h_dummy_qkv_weights = (T*)malloc(sizeof(T) * hidden_units * (head_num + 2 * kv_head_num) * head_size);
   // T* h_dummy_qkv_bias = (T*)malloc(sizeof(T) * (head_num + 2 * kv_head_num) * head_size);
    T* h_dummy_output_weights = (T*)malloc(sizeof(T) * hidden_units * hidden_units);
    T* h_dummy_output_bias = (T*)malloc(sizeof(T) * hidden_units);
    T* h_dummy_ffn_down = (T*)malloc(sizeof(T) * hidden_units * inter_size);
    T* h_dummy_ffn_down_bias = (T*)malloc(sizeof(T) * hidden_units);
    T* h_dummy_ffn_gate_up = (T*)malloc(sizeof(T) * hidden_units * 2 * inter_size);
    // T* h_dummy_ffn_up = (T*)malloc(sizeof(T) * hidden_units * inter_size);

    for (int i = 0; i < hidden_units; i++){
        h_dummy_attn_norm_weight[i] = (T)(rand() % 100 / (float)100000);
        h_dummy_ffn_norm_weight[i] = (T)(rand() % 100 / (float)100000);
        h_dummy_output_bias[i] = (T)(rand() % 100 / (float)100000);
        h_dummy_ffn_down_bias[i] = (T)(rand() % 100 / (float)100000);
    }
    //for (int i = 0; i < (head_num + 2 * kv_head_num) * head_size; i++) {
    //    h_dummy_qkv_bias[i] = (T)(rand() % 100 / (float)100000);
    //}
    for (int i = 0; i < hidden_units * inter_size; i++) {
        h_dummy_ffn_down[i] = (T)(rand() % 100 / (float)100000);
    }
    for (int i = 0; i < hidden_units * 2 * inter_size; i++) {   
        h_dummy_ffn_gate_up[i] = (T)(rand() % 100 / (float)100000);
        // h_dummy_ffn_up[i] = (T)1.0f;
    }
    for (int i = 0; i < hidden_units * hidden_units; i++) {
        h_dummy_output_weights[i] = (T)(rand() % 100 / (float)100000);
    }
    for (int i = 0; i < hidden_units * (head_num + 2 * kv_head_num) * head_size; i++) {
        h_dummy_qkv_weights[i] = (T)(rand() % 100 / (float)100000);
    }
    CHECK(cudaMemcpy(d_dummy_attn_norm_weight, h_dummy_attn_norm_weight, sizeof(T) * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_ffn_norm_weight, h_dummy_ffn_norm_weight, sizeof(T) * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_qkv_weights, h_dummy_qkv_weights, sizeof(T) * hidden_units * (head_num + 2 * kv_head_num) * head_size, cudaMemcpyHostToDevice));
    //CHECK(cudaMemcpy(d_dummy_qkv_bias, h_dummy_qkv_bias, sizeof(T) * (head_num + 2 * kv_head_num) * head_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_output_weights, h_dummy_output_weights, sizeof(T) * hidden_units * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_output_bias, h_dummy_output_bias, sizeof(T) * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_ffn_down, h_dummy_ffn_down, sizeof(T) * hidden_units * inter_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_ffn_down_bias, h_dummy_ffn_down_bias, sizeof(T) * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_ffn_gate_up, h_dummy_ffn_gate_up, sizeof(T) * hidden_units * 2 * inter_size, cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(d_dummy_ffn_up, h_dummy_ffn_up, sizeof(T) * hidden_units * inter_size, cudaMemcpyHostToDevice));
    // before kernel launch, the ptr is always void*, when luanching kernel, ptr type will be cast to float* or T*
    attn_norm_weight.gamma = d_dummy_attn_norm_weight;
    ffn_norm_weight.gamma = d_dummy_ffn_norm_weight;
    self_attn_weight.qkv.data = d_dummy_qkv_weights;
    self_attn_weight.qkv.bias = nullptr;
    self_attn_weight.output.data = d_dummy_output_weights;
    self_attn_weight.output.bias = d_dummy_output_bias;
    ffn_weight.gateAndup.data = d_dummy_ffn_gate_up;
    //ffn_weight.up.data = d_dummy_ffn_up;
    ffn_weight.down.data = d_dummy_ffn_down;
    ffn_weight.down.bias = d_dummy_ffn_down_bias;
}

template<typename T>
void freeWeights(BaseWeight<T>& weights)
{
    cudaFree(weights.data);
    if(weights.bias != nullptr) {
        cudaFree(weights.bias);
    }

    weights.data = nullptr;
    weights.bias = nullptr;
}
template<typename T>
LlamaLayerWeight<T>::~LlamaLayerWeight()
{
    // free norm weights ptr
    cudaFree(attn_norm_weight.gamma);
    cudaFree(ffn_norm_weight.gamma);
    // free other weights, including data and bias
    freeWeights(self_attn_weight.qkv);
    freeWeights(self_attn_weight.output);
    freeWeights(ffn_weight.gateAndup);
    // freeWeights(ffn_weight.up);
    freeWeights(ffn_weight.down);
}
// template instantial required in linking time
template class LlamaLayerWeight<float>;
template class LlamaLayerWeight<half>;
