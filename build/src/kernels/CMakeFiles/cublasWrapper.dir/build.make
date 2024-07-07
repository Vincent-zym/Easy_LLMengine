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
include src/kernels/CMakeFiles/cublasWrapper.dir/depend.make

# Include the progress variables for this target.
include src/kernels/CMakeFiles/cublasWrapper.dir/progress.make

# Include the compile flags for this target's objects.
include src/kernels/CMakeFiles/cublasWrapper.dir/flags.make

src/kernels/CMakeFiles/cublasWrapper.dir/cublas_utils.cc.o: src/kernels/CMakeFiles/cublasWrapper.dir/flags.make
src/kernels/CMakeFiles/cublasWrapper.dir/cublas_utils.cc.o: ../src/kernels/cublas_utils.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/workspace/Easy_LLMengine/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/kernels/CMakeFiles/cublasWrapper.dir/cublas_utils.cc.o"
	cd /root/workspace/Easy_LLMengine/build/src/kernels && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cublasWrapper.dir/cublas_utils.cc.o -c /root/workspace/Easy_LLMengine/src/kernels/cublas_utils.cc

src/kernels/CMakeFiles/cublasWrapper.dir/cublas_utils.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cublasWrapper.dir/cublas_utils.cc.i"
	cd /root/workspace/Easy_LLMengine/build/src/kernels && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/workspace/Easy_LLMengine/src/kernels/cublas_utils.cc > CMakeFiles/cublasWrapper.dir/cublas_utils.cc.i

src/kernels/CMakeFiles/cublasWrapper.dir/cublas_utils.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cublasWrapper.dir/cublas_utils.cc.s"
	cd /root/workspace/Easy_LLMengine/build/src/kernels && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/workspace/Easy_LLMengine/src/kernels/cublas_utils.cc -o CMakeFiles/cublasWrapper.dir/cublas_utils.cc.s

# Object files for target cublasWrapper
cublasWrapper_OBJECTS = \
"CMakeFiles/cublasWrapper.dir/cublas_utils.cc.o"

# External object files for target cublasWrapper
cublasWrapper_EXTERNAL_OBJECTS =

src/kernels/CMakeFiles/cublasWrapper.dir/cmake_device_link.o: src/kernels/CMakeFiles/cublasWrapper.dir/cublas_utils.cc.o
src/kernels/CMakeFiles/cublasWrapper.dir/cmake_device_link.o: src/kernels/CMakeFiles/cublasWrapper.dir/build.make
src/kernels/CMakeFiles/cublasWrapper.dir/cmake_device_link.o: src/kernels/CMakeFiles/cublasWrapper.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/workspace/Easy_LLMengine/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/cublasWrapper.dir/cmake_device_link.o"
	cd /root/workspace/Easy_LLMengine/build/src/kernels && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cublasWrapper.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/kernels/CMakeFiles/cublasWrapper.dir/build: src/kernels/CMakeFiles/cublasWrapper.dir/cmake_device_link.o

.PHONY : src/kernels/CMakeFiles/cublasWrapper.dir/build

# Object files for target cublasWrapper
cublasWrapper_OBJECTS = \
"CMakeFiles/cublasWrapper.dir/cublas_utils.cc.o"

# External object files for target cublasWrapper
cublasWrapper_EXTERNAL_OBJECTS =

lib/libcublasWrapper.a: src/kernels/CMakeFiles/cublasWrapper.dir/cublas_utils.cc.o
lib/libcublasWrapper.a: src/kernels/CMakeFiles/cublasWrapper.dir/build.make
lib/libcublasWrapper.a: src/kernels/CMakeFiles/cublasWrapper.dir/cmake_device_link.o
lib/libcublasWrapper.a: src/kernels/CMakeFiles/cublasWrapper.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/workspace/Easy_LLMengine/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library ../../lib/libcublasWrapper.a"
	cd /root/workspace/Easy_LLMengine/build/src/kernels && $(CMAKE_COMMAND) -P CMakeFiles/cublasWrapper.dir/cmake_clean_target.cmake
	cd /root/workspace/Easy_LLMengine/build/src/kernels && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cublasWrapper.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/kernels/CMakeFiles/cublasWrapper.dir/build: lib/libcublasWrapper.a

.PHONY : src/kernels/CMakeFiles/cublasWrapper.dir/build

src/kernels/CMakeFiles/cublasWrapper.dir/clean:
	cd /root/workspace/Easy_LLMengine/build/src/kernels && $(CMAKE_COMMAND) -P CMakeFiles/cublasWrapper.dir/cmake_clean.cmake
.PHONY : src/kernels/CMakeFiles/cublasWrapper.dir/clean

src/kernels/CMakeFiles/cublasWrapper.dir/depend:
	cd /root/workspace/Easy_LLMengine/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/workspace/Easy_LLMengine /root/workspace/Easy_LLMengine/src/kernels /root/workspace/Easy_LLMengine/build /root/workspace/Easy_LLMengine/build/src/kernels /root/workspace/Easy_LLMengine/build/src/kernels/CMakeFiles/cublasWrapper.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/kernels/CMakeFiles/cublasWrapper.dir/depend
