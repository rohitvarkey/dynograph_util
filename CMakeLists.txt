cmake_minimum_required (VERSION 2.8.11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_library(dynograph_util dynograph_util.cc)

find_package( OpenMP )
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(sort_bin "binary_sort.cc")
target_link_libraries(sort_bin dynograph_util)
# Enable parallel versions of std algorithms
target_compile_definitions(sort_bin PUBLIC _GLIBCXX_PARALLEL)

add_executable(dedup_bin "binary_dedup.cc")
target_link_libraries(dedup_bin dynograph_util)
# Enable parallel versions of std algorithms
target_compile_definitions(dedup_bin PUBLIC _GLIBCXX_PARALLEL)
