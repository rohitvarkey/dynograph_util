CXX_FLAGS += -std=c++11 -O3 -Wall -fopenmp -I.
CXX_FLAGS += -DENABLE_PERF_HOOKS

libhooks.a: hooks.o pfm_cxx.o
	ar rcs libhooks.a hooks.o pfm_cxx.o

%.o : %.cpp *.h
	$(CXX) -c $(CXX_FLAGS) $< -o $@$
