# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/workspace/Easy_LLMengine

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/workspace/Easy_LLMengine/build

# Include any dependencies generated for this target.
include src/kernels/CMakeFiles/qkv_bias_and_rope.dir/depend.make

# Include the progress variables for this target.
include src/kernels/CMakeFiles/qkv_bias_and_rope.dir/progress.make

# Include the compile flags for this target's objects.
include src/kernels/CMakeFiles/qkv_bias_and_rope.dir/flags.make

src/kernels/CMakeFiles/qkv_bias_and_rope.dir/qkv_bias_and_RoPE.cu.o: src/kernels/CMakeFiles/qkv_bias_and_rope.dir/flags.make
src/kernels/CMakeFiles/qkv_bias_and_rope.dir/qkv_bias_and_RoPE.cu.o: ../src/kernels/qkv_bias_and_RoPE.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/workspace/Easy_LLMengine/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object src/kernels/CMakeFiles/qkv_bias_and_rope.dir/qkv_bias_and_RoPE.cu.o"
	cd /root/workspace/Easy_LLMengine/build/src/kernels && /usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -dc /root/workspace/Easy_LLMengine/src/kernels/qkv_bias_and_RoPE.cu -o CMakeFiles/qkv_bias_and_rope.dir/qkv_bias_and_RoPE.cu.o

src/kernels/CMakeFiles/qkv_bias_and_rope.dir/qkv_bias_and_RoPE.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/qkv_bias_and_rope.dir/qkv_bias_and_RoPE.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

src/kernels/CMakeFiles/qkv_bias_and_rope.dir/qkv_bias_and_RoPE.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/qkv_bias_and_rope.dir/qkv_bias_and_RoPE.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target qkv_bias_and_rope
qkv_bias_and_rope_OBJECTS = \
"CMakeFiles/qkv_bias_and_rope.dir/qkv_bias_and_RoPE.cu.o"

# External object files for target qkv_bias_and_rope
qkv_bias_and_rope_EXTERNAL_OBJECTS =

src/kernels/CMakeFiles/qkv_bias_and_rope.dir/cmake_device_link.o: src/kernels/CMakeFiles/qkv_bias_and_rope.dir/qkv_bias_and_RoPE.cu.o
src/kernels/CMakeFiles/qkv_bias_and_rope.dir/cmake_device_link.o: src/kernels/CMakeFiles/qkv_bias_and_rope.dir/build.make
src/kernels/CMakeFiles/qkv_bias_and_rope.dir/cmake_device_link.o: src/kernels/CMakeFiles/qkv_bias_and_rope.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/workspace/Easy_LLMengine/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/qkv_bias_and_rope.dir/cmake_device_link.o"
	cd /root/workspace/Easy_LLMengine/build/src/kernels && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/qkv_bias_and_rope.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/kernels/CMakeFiles/qkv_bias_and_rope.dir/build: src/kernels/CMakeFiles/qkv_bias_and_rope.dir/cmake_device_link.o

.PHONY : src/kernels/CMakeFiles/qkv_bias_and_rope.dir/build

# Object files for target qkv_bias_and_rope
qkv_bias_and_rope_OBJECTS = \
"CMakeFiles/qkv_bias_and_rope.dir/qkv_bias_and_RoPE.cu.o"

# External object files for target qkv_bias_and_rope
qkv_bias_and_rope_EXTERNAL_OBJECTS =

lib/libqkv_bias_and_rope.a: src/kernels/CMakeFiles/qkv_bias_and_rope.dir/qkv_bias_and_RoPE.cu.o
lib/libqkv_bias_and_rope.a: src/kernels/CMakeFiles/qkv_bias_and_rope.dir/build.make
lib/libqkv_bias_and_rope.a: src/kernels/CMakeFiles/qkv_bias_and_rope.dir/cmake_device_link.o
lib/libqkv_bias_and_rope.a: src/kernels/CMakeFiles/qkv_bias_and_rope.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/workspace/Easy_LLMengine/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA static library ../../lib/libqkv_bias_and_rope.a"
	cd /root/workspace/Easy_LLMengine/build/src/kernels && $(CMAKE_COMMAND) -P CMakeFiles/qkv_bias_and_rope.dir/cmake_clean_target.cmake
	cd /root/workspace/Easy_LLMengine/build/src/kernels && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/qkv_bias_and_rope.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/kernels/CMakeFiles/qkv_bias_and_rope.dir/build: lib/libqkv_bias_and_rope.a

.PHONY : src/kernels/CMakeFiles/qkv_bias_and_rope.dir/build

src/kernels/CMakeFiles/qkv_bias_and_rope.dir/clean:
	cd /root/workspace/Easy_LLMengine/build/src/kernels && $(CMAKE_COMMAND) -P CMakeFiles/qkv_bias_and_rope.dir/cmake_clean.cmake
.PHONY : src/kernels/CMakeFiles/qkv_bias_and_rope.dir/clean

src/kernels/CMakeFiles/qkv_bias_and_rope.dir/depend:
	cd /root/workspace/Easy_LLMengine/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/workspace/Easy_LLMengine /root/workspace/Easy_LLMengine/src/kernels /root/workspace/Easy_LLMengine/build /root/workspace/Easy_LLMengine/build/src/kernels /root/workspace/Easy_LLMengine/build/src/kernels/CMakeFiles/qkv_bias_and_rope.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/kernels/CMakeFiles/qkv_bias_and_rope.dir/depend

