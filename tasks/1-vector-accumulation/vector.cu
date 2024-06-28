#include "vector.cuh"
#include <cassert>
#include <cstdio>
#include <vector>

__global__ void Accumulate(float* a, std::size_t n, std::size_t dim, float* b) {
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	if(j < dim) {
		b[j] = 0;
		for(int i = 0; i < n; ++i) b[j] += a[i * dim + j];
	}
}
