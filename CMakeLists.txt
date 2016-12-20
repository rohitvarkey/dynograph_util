cmake_minimum_required (VERSION 2.8.11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_subdirectory(hooks)

if (MPI_FOUND)
  include_directories(${MPI_CXX_INCLUDE_PATH})
  add_definitions(-DUSE_MPI)
endif()

find_package( OpenMP )
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Build the dynograph_util library
add_library(dynograph_util dynograph_util.cc)
target_compile_definitions(dynograph_util PUBLIC _GLIBCXX_PARALLEL)
target_link_libraries(dynograph_util hooks)
target_include_directories(dynograph_util PUBLIC hooks)

# Detect if googletest was already built elsewhere
if (NOT GOOGLETEST_DIR)
  set(GOOGLETEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/googletest/include PARENT_SCOPE)
  add_subdirectory(googletest)
endif()

# Build the unit tests
add_executable(dynograph_util_test dynograph_util_test.cc)
target_include_directories(dynograph_util_test PRIVATE ${GOOGLETEST_DIR})
target_link_libraries(dynograph_util_test dynograph_util gtest_main)

add_test(dynograph_util_test dynograph_util_test)