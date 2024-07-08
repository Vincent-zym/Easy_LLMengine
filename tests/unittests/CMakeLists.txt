add_executable(embedding
    test_input_embedding.cu
)
target_link_libraries(
    embedding PUBLIC
    -lcudart
    -lcudadevrt
    embeddingFunctor)

add_executable(paddingoffset
    test_cal_paddingoffset.cu
)
target_link_libraries(
  paddingoffset PUBLIC
    -lcudart
    -lcudadevrt
    cal_paddingoffset) 

add_executable(rms_norm
    test_rmsnorm.cu
)
target_link_libraries(
    rms_norm PUBLIC
    -lcudart
    -lcudadevrt
    rmsnorm)

add_executable(causalmask
    test_casual_mask.cu
)
target_link_libraries(  # Libs for test_build_casual_mask
   causalmask PUBLIC
    -lcudart
    -lcudadevrt
    build_casual_mask) 

add_executable(testlinear
    test_linear.cu
)
target_link_libraries(  # Libs for test_build_casual_mask
    testlinear PUBLIC
    -lcudart
    -lcudadevrt
    linear) 

add_executable(biasRope
    test_bias_and_RoPE.cu
)
target_link_libraries(  # Libs for test_qkv_bias_and_rope
  biasRope PUBLIC
    -lcudart
    -lcudadevrt
    qkv_bias_and_rope)

add_executable(test_concat_kv
    test_concat_kv.cu
)
target_link_libraries(  # Libs for test_qkv_bias_and_rope
    test_concat_kv PUBLIC
    -lcudart
    -lcudadevrt
    concat_kv) 
