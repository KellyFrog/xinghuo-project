#include "matrix.cuh"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cassert>

int main() {
	std::size_t n = 1 << 9;
	std::size_t m = 3 << 8 | 5;
	std::size_t k = 5 << 9 | 9;
	std::size_t runs = 1 << 5;

	float* a = (float*)std::malloc(n * m * sizeof(float));
	float* b = (float*)std::malloc(m * k * sizeof(float));
	float* c = (float*)std::malloc(n * k * sizeof(float));
	float* d = (float*)std::malloc(n * k * sizeof(float));
	std::mt19937 rnd(0x114514);
	for(int i = 0; i < n; ++i) {
		for(int j = 0; j < m; ++j) {
			a[i * m + j] = rnd() / 1e7;
		}
	}
	for(int i = 0; i < m; ++i) {
		for(int j = 0; j < k; ++j) {
			b[i * k + j] = rnd() / 1e7;
		}
	}
	float* d_a = NULL;
	float* d_b = NULL;
	float* d_c = NULL;
	assert(cudaMalloc((void**) &d_a, n * m * sizeof(float)) == cudaSuccess);
	assert(cudaMalloc((void**) &d_b, m * k * sizeof(float)) == cudaSuccess);
	assert(cudaMalloc((void**) &d_c, n * k * sizeof(float)) == cudaSuccess);
	assert(cudaMemcpy(d_a, a, n * m * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
	assert(cudaMemcpy(d_b, b, m * k * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);

	cudaEvent_t startTime = 0, endTime = 0;
	cudaEventCreate(&startTime);
	cudaEventCreate(&endTime);
	cudaEventRecord(startTime, 0);
	cudaEventSynchronize(startTime);
	for(int t = 0; t < runs; ++t) {
		/*
		*/
		MatrixMul(d_a, d_b, n, m, k, d_c);
	}
	cudaEventRecord(endTime, 0);
	cudaEventSynchronize(endTime);
	float gpuTime = 0;
	cudaEventElapsedTime(&gpuTime, startTime, endTime);
	printf("GPU: %lu runs, n = %lu, m = %lu, k = %lu, total time = %fms, arv time = %fms\n", runs, n, m, k, gpuTime, gpuTime / runs);
	for(int i = 0; i < n; ++i) {
		for(int j = 0; j < k; ++j) {
			float res = 0;
			for(int p = 0; p < m; ++p) {
				res += a[i * m + p] * b[p * k + j];
			}
			c[i * k + j] = res;
		}
	}
	assert(cudaMemcpy(d, d_c, n * k * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
	for(int i = 0; i < n; ++i) {
		for(int j = 0; j < k; ++j) {
			assert(fabs(d[i * k + j] - c[i * k + j]) / std::max(1.0f, fabs(d[i * k + j])) < 1e-3);
		}
	}
	assert(cudaGetLastError() == cudaSuccess);
	assert(cudaFree(d_a) == cudaSuccess);
	assert(cudaFree(d_b) == cudaSuccess);
	assert(cudaFree(d_c) == cudaSuccess);
	std::free(a);
	std::free(b);
	std::free(c);

	return 0;
}
