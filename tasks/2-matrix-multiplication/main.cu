#include "matrix.cuh"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cassert>

int main() {
	size_t n = 1 << 9;
	size_t m = 3 << 8 | 5;
	size_t k = 5 << 7 | 9;
	float* a = (float*)std::malloc(n * m * sizeof(float));
	float* b = (float*)std::malloc(m * k * sizeof(float));
	float* c = (float*)std::malloc(n * k * sizeof(float));
	float* d = (float*)std::malloc(n * k * sizeof(float));
	std::mt19937 rnd(0x114514);
	for(int i = 0; i < n; ++i) {
		for(int j = 0; j < m; ++j) {
			a[i * m + j] = rnd() / 1e7;
			//a[i * m + j] = (i == j);
		}
	}
	for(int i = 0; i < m; ++i) {
		for(int j = 0; j < k; ++j) {
			b[i * k + j] = rnd() / 1e7;
			//b[i * k + j] = (i == j);
		}
	}
	float* device_a = NULL;
	float* device_b = NULL;
	float* device_c = NULL;
	assert(cudaMalloc((void**) &device_a, n * m * sizeof(float)) == cudaSuccess);
	assert(cudaMalloc((void**) &device_b, m * k * sizeof(float)) == cudaSuccess);
	assert(cudaMalloc((void**) &device_c, n * k * sizeof(float)) == cudaSuccess);
	assert(cudaMemcpy(device_a, a, n * m * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
	assert(cudaMemcpy(device_b, b, m * k * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
	for(int t = 0; t < (1 << 5); ++t) {
		dim3 block(BSIZE, BSIZE);
		dim3 numBlock(n / BSIZE + 1, k / BSIZE + 1);
		MatrixMul<<<numBlock, block>>>(device_a, device_b, n, m, k, device_c);
	}
	for(int i = 0; i < n; ++i) {
		for(int j = 0; j < k; ++j) {
			float res = 0;
			for(int p = 0; p < m; ++p) {
				res += a[i * m + p] * b[p * k + j];
			}
			c[i * k + j] = res;
		}
	}
	assert(cudaMemcpy(d, device_c, n * k * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
	for(int i = 0; i < n; ++i) {
		for(int j = 0; j < k; ++j) {
			assert(fabs(d[i * k + j] - c[i * k + j]) / std::max(1.0f, fabs(d[i * k + j])) < 1e-3);
		}
	}
	assert(cudaGetLastError() == cudaSuccess);
	assert(cudaFree(device_a) == cudaSuccess);
	assert(cudaFree(device_b) == cudaSuccess);
	assert(cudaFree(device_c) == cudaSuccess);
	std::free(a);
	std::free(b);
	std::free(c);

	return 0;
}
