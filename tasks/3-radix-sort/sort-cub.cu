#include "sort.cuh"
#include <cub/cub.cuh>

void RadixSort(unsigned* d_a, std::size_t n) {
	unsigned* d_b = NULL;
	assert(cudaMalloc((void**)&d_b, n * sizeof(unsigned)) == cudaSuccess);
	void* d_temp = NULL;
	std::size_t tempSize = 0;
	cub::DeviceRadixSort::SortKeys(d_temp, tempSize, d_a, d_b, n);
	assert(cudaMalloc((void**)&d_temp, tempSize) == cudaSuccess);
	cub::DeviceRadixSort::SortKeys(d_temp, tempSize, d_a, d_b, n);
	assert(cudaMemcpy(d_a, d_b, n * sizeof(unsigned), cudaMemcpyDeviceToDevice) == cudaSuccess);
	assert(cudaFree(d_b) == cudaSuccess);
}
