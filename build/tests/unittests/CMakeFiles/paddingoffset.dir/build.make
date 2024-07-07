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
include tests/unittests/CMakeFiles/paddingoffset.dir/depend.make

# Include the progress variables for this target.
include tests/unittests/CMakeFiles/paddingoffset.dir/progress.make

# Include the compile flags for this target's objects.
include tests/unittests/CMakeFiles/paddingoffset.dir/flags.make

tests/unittests/CMakeFiles/paddingoffset.dir/test_cal_paddingoffset.cu.o: tests/unittests/CMakeFiles/paddingoffset.dir/flags.make
tests/unittests/CMakeFiles/paddingoffset.dir/test_cal_paddingoffset.cu.o: ../tests/unittests/test_cal_paddingoffset.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/workspace/Easy_LLMengine/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object tests/unittests/CMakeFiles/paddingoffset.dir/test_cal_paddingoffset.cu.o"
	cd /root/workspace/Easy_LLMengine/build/tests/unittests && /usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /root/workspace/Easy_LLMengine/tests/unittests/test_cal_paddingoffset.cu -o CMakeFiles/paddingoffset.dir/test_cal_paddingoffset.cu.o

tests/unittests/CMakeFiles/paddingoffset.dir/test_cal_paddingoffset.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/paddingoffset.dir/test_cal_paddingoffset.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

tests/unittests/CMakeFiles/paddingoffset.dir/test_cal_paddingoffset.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/paddingoffset.dir/test_cal_paddingoffset.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target paddingoffset
paddingoffset_OBJECTS = \
"CMakeFiles/paddingoffset.dir/test_cal_paddingoffset.cu.o"

# External object files for target paddingoffset
paddingoffset_EXTERNAL_OBJECTS =

bin/paddingoffset: tests/unittests/CMakeFiles/paddingoffset.dir/test_cal_paddingoffset.cu.o
bin/paddingoffset: tests/unittests/CMakeFiles/paddingoffset.dir/build.make
bin/paddingoffset: lib/libcal_paddingoffset.a
bin/paddingoffset: tests/unittests/CMakeFiles/paddingoffset.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/workspace/Easy_LLMengine/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable ../../bin/paddingoffset"
	cd /root/workspace/Easy_LLMengine/build/tests/unittests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/paddingoffset.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/unittests/CMakeFiles/paddingoffset.dir/build: bin/paddingoffset

.PHONY : tests/unittests/CMakeFiles/paddingoffset.dir/build

tests/unittests/CMakeFiles/paddingoffset.dir/clean:
	cd /root/workspace/Easy_LLMengine/build/tests/unittests && $(CMAKE_COMMAND) -P CMakeFiles/paddingoffset.dir/cmake_clean.cmake
.PHONY : tests/unittests/CMakeFiles/paddingoffset.dir/clean

tests/unittests/CMakeFiles/paddingoffset.dir/depend:
	cd /root/workspace/Easy_LLMengine/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/workspace/Easy_LLMengine /root/workspace/Easy_LLMengine/tests/unittests /root/workspace/Easy_LLMengine/build /root/workspace/Easy_LLMengine/build/tests/unittests /root/workspace/Easy_LLMengine/build/tests/unittests/CMakeFiles/paddingoffset.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/unittests/CMakeFiles/paddingoffset.dir/depend

