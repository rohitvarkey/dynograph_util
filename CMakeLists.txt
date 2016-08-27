cmake_minimum_required (VERSION 2.8.11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_library(dynograph_util dynograph_util.cc)

add_executable(sort_bin "binary_sort.cc")
target_link_libraries(sort_bin dynograph_util)
