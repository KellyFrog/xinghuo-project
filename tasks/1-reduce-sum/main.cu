#include <iostream>
#include <random>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include "sum.cuh"

int main() {
	std::mt19937 rnd(0x114514);

	std::size_t n = 1 << 24;
	std::size_t dim = 1 << 4;
	std::size_t runs = 1 << 5;
	unsigned* a = (unsigned*)malloc(n * dim * sizeof(unsigned));
	for(int i = 0; i < n * dim; ++i) {
		a[i] = rnd() / 1e5;
		//a[i] = 1;
	}
	unsigned* b = (unsigned*)malloc(dim * sizeof(unsigned));
	unsigned* c = (unsigned*)malloc(dim * sizeof(unsigned));
	std::memset(b, 0, dim * sizeof(unsigned));

	unsigned* device_a = NULL, *device_b = NULL;
	assert(cudaMalloc((void**) &device_a, n * dim * sizeof(unsigned)) == cudaSuccess);
	assert(cudaMalloc((void**) &device_b, dim * sizeof(unsigned)) == cudaSuccess);
	assert(cudaMemcpy(device_a, a, n * dim * sizeof(unsigned), cudaMemcpyHostToDevice) == cudaSuccess);
	assert(cudaMemset(device_b, 0, dim * sizeof(unsigned)) == cudaSuccess);

	cudaEvent_t stime = 0, etime = 0;
	cudaEventCreate(&stime);
	cudaEventCreate(&etime);
	cudaEventRecord(stime, 0);
	for(int t = 0; t < runs; ++t) {
		ReduceSum(device_a, n, dim, device_b);
	}
	cudaEventRecord(etime, 0);
	cudaEventSynchronize(etime);

	float gtime;
	cudaEventElapsedTime(&gtime, stime, etime);
	printf("GPU: %u runs, n = %u, dim = %u, total time = %fms, arv time = %fms\n", runs, n, dim, gtime, gtime / runs);

	cudaEventCreate(&stime);
	cudaEventCreate(&etime);
	cudaEventRecord(stime, 0);
	for(int t = 0; t < runs; ++t) {
		std::memset(b, 0, dim * sizeof(unsigned));
		for(int i = 0; i < n; ++i) {
			for(int j = 0; j < dim; ++j) {
				b[j] += a[i * dim + j];
			}
		}
	}
	cudaEventRecord(etime, 0);
	cudaEventSynchronize(etime);

	float ctime;
	cudaEventElapsedTime(&ctime, stime, etime);
	printf("CPU: %u runs, n = %u, dim = %u, total time = %fms, arv time = %fms\n", runs, n, dim, ctime, ctime / runs);

	assert(cudaMemcpy(c, device_b, dim * sizeof(unsigned), cudaMemcpyDeviceToHost) == cudaSuccess);

	std::cerr << std::fixed << std::setprecision(15);
	for(int i = 0; i < dim; ++i) std::cerr << c[i] << " \n"[i == dim - 1];
	for(int i = 0; i < dim; ++i) std::cerr << b[i] << " \n"[i == dim - 1];

	//for(int i = 0; i < dim; ++i) assert(std::fabs(c[i] - b[i]) / std::max(fabs(b[i]), 1.0f) < 1e-3);
	for(int i = 0; i < dim; ++i) assert(c[i] == b[i]);

	assert(cudaFree(device_a) == cudaSuccess);
	assert(cudaFree(device_b) == cudaSuccess);
	free(a);
	free(b);

	return 0;
}
