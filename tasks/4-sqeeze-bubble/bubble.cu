#include "bubble.cuh"

#include <cstdlib>
#include <cassert>
#include <cstdio>

const int BSIZE = 32;
const int THREADS = BSIZE * BSIZE;

__global__ void MakePrefix(unsigned* a, std::size_t n, std::size_t size, int* cnt) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int l = id * size, r = (id + 1) * size;
	if(r > n) r = n;
	int c = 0;
	for(int i = l; i < r; ++i) {
		c += !!a[i];
	}
	cnt[id] = c;
	__syncthreads();
	if(id == 0) {
		for(int i = 1; i < THREADS; ++i) cnt[i] += cnt[i-1];
	}
}

__global__ void MoveForward(unsigned* a, std::size_t n, std::size_t size, int* cnt, unsigned* b) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int l = id * size, r = (id + 1) * size;
	if(r > n) r = n;
	int c = cnt[id];
	for(int i = r - 1; i >= l; --i) if(a[i]) b[--c] = a[i];
}

int SqeezeBubble(unsigned* a, std::size_t n) {
	std::size_t size = (n + THREADS - 1) / THREADS;
	int* cnt = NULL;
	assert(cudaMalloc((void**)&cnt, THREADS * sizeof(int)) == cudaSuccess);
	assert(cudaMemset(cnt, 0, THREADS * sizeof(int)) == cudaSuccess);

	MakePrefix<<<BSIZE, BSIZE>>>(a, n, size, cnt);
	int r;
	assert(cudaMemcpy(&r, cnt + THREADS - 1, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
	unsigned* b = NULL;
	assert(cudaMalloc((void**)&b, r * sizeof(unsigned)) == cudaSuccess);
	MoveForward<<<BSIZE, BSIZE>>>(a, n, size, cnt, b);
	assert(cudaMemcpy(a, b, r * sizeof(unsigned), cudaMemcpyDeviceToDevice) == cudaSuccess);
	assert(cudaFree(cnt) == cudaSuccess);
	assert(cudaFree(b) == cudaSuccess);
	return r;
}
