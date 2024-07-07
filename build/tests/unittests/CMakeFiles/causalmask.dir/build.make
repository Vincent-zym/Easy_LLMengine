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
include tests/unittests/CMakeFiles/causalmask.dir/depend.make

# Include the progress variables for this target.
include tests/unittests/CMakeFiles/causalmask.dir/progress.make

# Include the compile flags for this target's objects.
include tests/unittests/CMakeFiles/causalmask.dir/flags.make

tests/unittests/CMakeFiles/causalmask.dir/test_casual_mask.cu.o: tests/unittests/CMakeFiles/causalmask.dir/flags.make
tests/unittests/CMakeFiles/causalmask.dir/test_casual_mask.cu.o: ../tests/unittests/test_casual_mask.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/workspace/Easy_LLMengine/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object tests/unittests/CMakeFiles/causalmask.dir/test_casual_mask.cu.o"
	cd /root/workspace/Easy_LLMengine/build/tests/unittests && /usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /root/workspace/Easy_LLMengine/tests/unittests/test_casual_mask.cu -o CMakeFiles/causalmask.dir/test_casual_mask.cu.o

tests/unittests/CMakeFiles/causalmask.dir/test_casual_mask.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/causalmask.dir/test_casual_mask.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

tests/unittests/CMakeFiles/causalmask.dir/test_casual_mask.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/causalmask.dir/test_casual_mask.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target causalmask
causalmask_OBJECTS = \
"CMakeFiles/causalmask.dir/test_casual_mask.cu.o"

# External object files for target causalmask
causalmask_EXTERNAL_OBJECTS =

bin/causalmask: tests/unittests/CMakeFiles/causalmask.dir/test_casual_mask.cu.o
bin/causalmask: tests/unittests/CMakeFiles/causalmask.dir/build.make
bin/causalmask: lib/libbuild_casual_mask.a
bin/causalmask: tests/unittests/CMakeFiles/causalmask.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/workspace/Easy_LLMengine/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable ../../bin/causalmask"
	cd /root/workspace/Easy_LLMengine/build/tests/unittests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/causalmask.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/unittests/CMakeFiles/causalmask.dir/build: bin/causalmask

.PHONY : tests/unittests/CMakeFiles/causalmask.dir/build

tests/unittests/CMakeFiles/causalmask.dir/clean:
	cd /root/workspace/Easy_LLMengine/build/tests/unittests && $(CMAKE_COMMAND) -P CMakeFiles/causalmask.dir/cmake_clean.cmake
.PHONY : tests/unittests/CMakeFiles/causalmask.dir/clean

tests/unittests/CMakeFiles/causalmask.dir/depend:
	cd /root/workspace/Easy_LLMengine/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/workspace/Easy_LLMengine /root/workspace/Easy_LLMengine/tests/unittests /root/workspace/Easy_LLMengine/build /root/workspace/Easy_LLMengine/build/tests/unittests /root/workspace/Easy_LLMengine/build/tests/unittests/CMakeFiles/causalmask.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/unittests/CMakeFiles/causalmask.dir/depend
