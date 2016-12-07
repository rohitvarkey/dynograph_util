cmake_minimum_required (VERSION 2.8.11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_subdirectory(hooks)

add_library(dynograph_util STATIC dynograph_util.cc)
target_compile_definitions(dynograph_util PUBLIC _GLIBCXX_PARALLEL)
target_link_libraries(hooks)
target_include_directories(dynograph_util PRIVATE hooks)

find_package( OpenMP )
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
