cmake_minimum_required (VERSION 2.8.11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_subdirectory(hooks)

# Build the dynograph_util library
add_library(dynograph_util dynograph_util.cc)
# Enable parallel versions of functions from <algorithm> and <numeric>
if (OPENMP_FOUND)
  target_compile_definitions(dynograph_util PUBLIC _GLIBCXX_PARALLEL)
endif()
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
target_link_libraries(dynograph_util_test dynograph_util gtest)

add_executable(reference_impl_test reference_impl.cc)
target_include_directories(reference_impl_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${GOOGLETEST_DIR})
target_link_libraries(reference_impl_test dynograph_util gtest_main)

enable_testing()
add_test(dynograph_util_test dynograph_util_test ${CMAKE_CURRENT_SOURCE_DIR}/data/ring-of-cliques.graph.el)
add_test(reference_impl_test dynograph_util_test)

# Copy test data to the build directory
file(
    COPY
    data/ring-of-cliques.graph.bin
    data/worldcup-10K.graph.bin
    DESTINATION
    data/
)
