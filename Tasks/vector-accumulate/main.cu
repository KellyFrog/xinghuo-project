#include <iostream>
#include <random>
#include <cassert>
#include <cmath>
#include <cstring>
#include "vector.cuh"

const int THREADS = 1024;

int main() {
	std::mt19937 rnd(0x114514);

	std::size_t n = 1 << 10;
	std::size_t dim = 1 << 15;
	float* a = (float*)malloc(n * dim * sizeof(float));
	for(int i = 0; i < n * dim; ++i) {
		a[i] = rnd() / 1e5;
		//a[i] = 1;
	}
	float* b = (float*)malloc(dim * sizeof(float));
	float* c = (float*)malloc(dim * sizeof(float));
	std::memset(b, 0, dim * sizeof(float));

	float* device_a = NULL, *device_b = NULL;
	assert(cudaMalloc((void**) &device_a, n * dim * sizeof(float)) == cudaSuccess);
	assert(cudaMalloc((void**) &device_b, dim * sizeof(float)) == cudaSuccess);
	assert(cudaMemcpy(device_a, a, n * dim * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
	assert(cudaMemset(device_b, 0, dim * sizeof(float)) == cudaSuccess);

	for(int t = 0; t < (1 << 10); ++t) {
		Accumulate<<<dim / THREADS + 1, THREADS>>>(device_a, n, dim, device_b);
	}

	for(int i = 0; i < n; ++i) {
		for(int j = 0; j < dim; ++j) {
			b[j] += a[i * dim + j];
		}
	}

	assert(cudaMemcpy(c, device_b, dim * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);

	for(int i = 0; i < dim; ++i) assert(std::fabs(c[i] - b[i]) / std::max(fabs(b[i]), 1.0f) < 1e-3);

	assert(cudaFree(device_a) == cudaSuccess);
	assert(cudaFree(device_b) == cudaSuccess);
	free(a);
	free(b);

	return 0;
}
