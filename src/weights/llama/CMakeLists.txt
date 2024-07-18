add_library(layerweights STATIC layer_weights.cc)
target_link_libraries(layerweights PUBLIC -lcudart weightutils)
