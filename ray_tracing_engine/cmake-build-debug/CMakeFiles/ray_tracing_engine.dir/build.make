# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_SOURCE_DIR = /mnt/c/Users/mySab/Documents/SourceCode/Git-Repository/Ray-Tracing-Engine/ray_tracing_engine

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/c/Users/mySab/Documents/SourceCode/Git-Repository/Ray-Tracing-Engine/ray_tracing_engine/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/ray_tracing_engine.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ray_tracing_engine.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ray_tracing_engine.dir/flags.make

CMakeFiles/ray_tracing_engine.dir/main.cpp.o: CMakeFiles/ray_tracing_engine.dir/flags.make
CMakeFiles/ray_tracing_engine.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/mySab/Documents/SourceCode/Git-Repository/Ray-Tracing-Engine/ray_tracing_engine/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ray_tracing_engine.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ray_tracing_engine.dir/main.cpp.o -c /mnt/c/Users/mySab/Documents/SourceCode/Git-Repository/Ray-Tracing-Engine/ray_tracing_engine/main.cpp

CMakeFiles/ray_tracing_engine.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ray_tracing_engine.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/mySab/Documents/SourceCode/Git-Repository/Ray-Tracing-Engine/ray_tracing_engine/main.cpp > CMakeFiles/ray_tracing_engine.dir/main.cpp.i

CMakeFiles/ray_tracing_engine.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ray_tracing_engine.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/mySab/Documents/SourceCode/Git-Repository/Ray-Tracing-Engine/ray_tracing_engine/main.cpp -o CMakeFiles/ray_tracing_engine.dir/main.cpp.s

CMakeFiles/ray_tracing_engine.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/ray_tracing_engine.dir/main.cpp.o.requires

CMakeFiles/ray_tracing_engine.dir/main.cpp.o.provides: CMakeFiles/ray_tracing_engine.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/ray_tracing_engine.dir/build.make CMakeFiles/ray_tracing_engine.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/ray_tracing_engine.dir/main.cpp.o.provides

CMakeFiles/ray_tracing_engine.dir/main.cpp.o.provides.build: CMakeFiles/ray_tracing_engine.dir/main.cpp.o


# Object files for target ray_tracing_engine
ray_tracing_engine_OBJECTS = \
"CMakeFiles/ray_tracing_engine.dir/main.cpp.o"

# External object files for target ray_tracing_engine
ray_tracing_engine_EXTERNAL_OBJECTS =

ray_tracing_engine: CMakeFiles/ray_tracing_engine.dir/main.cpp.o
ray_tracing_engine: CMakeFiles/ray_tracing_engine.dir/build.make
ray_tracing_engine: CMakeFiles/ray_tracing_engine.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/c/Users/mySab/Documents/SourceCode/Git-Repository/Ray-Tracing-Engine/ray_tracing_engine/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ray_tracing_engine"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ray_tracing_engine.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ray_tracing_engine.dir/build: ray_tracing_engine

.PHONY : CMakeFiles/ray_tracing_engine.dir/build

CMakeFiles/ray_tracing_engine.dir/requires: CMakeFiles/ray_tracing_engine.dir/main.cpp.o.requires

.PHONY : CMakeFiles/ray_tracing_engine.dir/requires

CMakeFiles/ray_tracing_engine.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ray_tracing_engine.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ray_tracing_engine.dir/clean

CMakeFiles/ray_tracing_engine.dir/depend:
	cd /mnt/c/Users/mySab/Documents/SourceCode/Git-Repository/Ray-Tracing-Engine/ray_tracing_engine/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/Users/mySab/Documents/SourceCode/Git-Repository/Ray-Tracing-Engine/ray_tracing_engine /mnt/c/Users/mySab/Documents/SourceCode/Git-Repository/Ray-Tracing-Engine/ray_tracing_engine /mnt/c/Users/mySab/Documents/SourceCode/Git-Repository/Ray-Tracing-Engine/ray_tracing_engine/cmake-build-debug /mnt/c/Users/mySab/Documents/SourceCode/Git-Repository/Ray-Tracing-Engine/ray_tracing_engine/cmake-build-debug /mnt/c/Users/mySab/Documents/SourceCode/Git-Repository/Ray-Tracing-Engine/ray_tracing_engine/cmake-build-debug/CMakeFiles/ray_tracing_engine.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ray_tracing_engine.dir/depend

