cmake_minimum_required(VERSION 3.8)
# 编译为静态库
add_library(Llamaffn STATIC ffn.cpp)
set_property(TARGET Llamaffn PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Llamaffn PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
# 链接Llamaffn所调用到函数对应的静态库
target_link_libraries(Llamaffn PUBLIC
                             -lcudart
                             -lcudadevrt
                             act
                             linear)