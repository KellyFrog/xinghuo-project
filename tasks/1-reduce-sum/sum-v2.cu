#include "sum.cuh"
#include <cassert>
#include <cstdio>
#include <vector>

const int BSIZE = 16;

__device__ unsigned d_sum[BSIZE][BSIZE];

__global__ void KReduceSumLoop(unsigned* a, std::size_t n, std::size_t dim, unsigned* b) {
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	if(j < dim) {
		unsigned sum = 0;
#pragma unroll
		for(int i = 0; i < n; ++i) sum += a[i * dim + j];
		b[j] = sum;
	}
}

__global__ void KInitShared() {
	d_sum[threadIdx.x][threadIdx.y] = 0;
}

__global__ void KAddToShared(unsigned* a, std::size_t n, std::size_t dim) {
	int i = (BSIZE / dim) * (threadIdx.x + blockIdx.x * blockDim.x) + threadIdx.y / dim;
	if(i < n) {
		int j = threadIdx.y % dim;
		atomicAdd(&d_sum[threadIdx.x][threadIdx.y], a[i * dim + j]);
	}
}

__global__ void KAddToResult(std::size_t n, std::size_t dim, unsigned* b) {
	atomicAdd(b + threadIdx.y % dim, d_sum[threadIdx.x][threadIdx.y]);
}

void ReduceSum(unsigned* a, std::size_t n, std::size_t dim, unsigned* b) {
	if(dim >= 128) {
		int threads = BSIZE * BSIZE;
		KReduceSumLoop<<<(n + threads - 1) / threads, threads>>>(a, n, dim, b);
	} else {
		assert(BSIZE % dim == 0);
		KInitShared<<<1, dim3(BSIZE, BSIZE)>>>();
		KAddToShared<<<(n * dim + BSIZE * BSIZE - 1) / (BSIZE * BSIZE), dim3(BSIZE, BSIZE)>>>(a, n, dim);
		assert(cudaMemset(b, 0, dim * sizeof(unsigned)) == cudaSuccess);
		KAddToResult<<<1, dim3(BSIZE, BSIZE)>>>(n, dim, b);
	}
}
