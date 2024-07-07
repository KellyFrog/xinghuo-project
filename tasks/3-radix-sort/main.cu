#include "sort.cuh"

#include <random>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>
// #include <thrust/sort.h>
// #include <thrust/device_vector.h>

int main(int argc, char** argv) {
	assert(argc == 4);
	std::size_t n = std::atoi(argv[1]);
	std::size_t t = std::atoi(argv[2]);
	std::size_t k = std::atoi(argv[3]);
	unsigned *a = (unsigned*) std::malloc(n * sizeof(unsigned));
	unsigned *b = (unsigned*) std::malloc(n * sizeof(unsigned));
	unsigned *d_a = NULL;
	std::mt19937 rnd(0x20061013);
	unsigned mask = (1u << k) - 1;
	for(int i = 0; i < n; ++i) b[i] = a[i] = rnd();
	assert(cudaMalloc((void**)&d_a, n * sizeof(unsigned)) == cudaSuccess);
	assert(cudaMemcpy(d_a, a, n * sizeof(unsigned), cudaMemcpyHostToDevice) == cudaSuccess);
	//thrust::device_vector<unsigned> d_vector(std::vector<int>(a, a + n));
	for(int i = 0; i < t; ++i) {
		RadixSort(d_a, n);
		//std::sort(b, b + n);
		//thrust::sort(d_vector.begin(), d_vector.end());
	}
	std::sort(b, b + n);
	assert(cudaMemcpy(a, d_a, n * sizeof(unsigned), cudaMemcpyDeviceToHost) == cudaSuccess);
	for(int i = 0; i < n; ++i) assert(a[i] == b[i]);
	std::free(a);
	std::free(b);
	assert(cudaFree(d_a) == cudaSuccess);

	return 0;
}
