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
	assert(argc == 3);
	std::size_t n = std::atoi(argv[1]);
	//std::size_t n; std::cin >> n;
	std::size_t runs = std::atoi(argv[2]);
	unsigned *a = (unsigned*) std::malloc(n * sizeof(unsigned));
	unsigned *b = (unsigned*) std::malloc(n * sizeof(unsigned));
	//for(int i = 0; i < n; ++i) std::cin >> a[i], b[i] = a[i];
	unsigned *d_a = NULL;
	std::mt19937 rnd(0x20061013);
	for(int i = 0; i < n; ++i) b[i] = a[i] = rnd();
	assert(cudaMalloc((void**)&d_a, n * sizeof(unsigned)) == cudaSuccess);
	assert(cudaMemcpy(d_a, a, n * sizeof(unsigned), cudaMemcpyHostToDevice) == cudaSuccess);
	cudaEvent_t startTime = 0, endTime = 0;
	cudaEventCreate(&startTime);
	cudaEventCreate(&endTime);
	cudaEventRecord(startTime, 0);
	cudaEventSynchronize(startTime);
	for(int i = 0; i < runs; ++i) {
		RadixSort(d_a, n);
	}
	cudaEventRecord(endTime, 0);
	cudaEventSynchronize(endTime);
	float gpuTime = 0;
	cudaEventElapsedTime(&gpuTime, startTime, endTime);
	printf("GPU: %lu runs, n = %lu, total time = %fms, arv time = %fms\n", runs, n, gpuTime, gpuTime / runs);
	std::sort(b, b + n);
	assert(cudaMemcpy(a, d_a, n * sizeof(unsigned), cudaMemcpyDeviceToHost) == cudaSuccess);
	for(int i = 0; i < n; ++i) assert(a[i] == b[i]);
	std::free(a);
	std::free(b);
	assert(cudaFree(d_a) == cudaSuccess);

	return 0;
}
