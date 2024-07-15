#include "sum.cuh"
#include <cassert>
#include <cstdio>
#include <vector>

const int BSIZE = 16;

__global__ void KReduceSumLoop(unsigned* a, std::size_t n, std::size_t dim, unsigned* b) {
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	if(j < dim) {
		unsigned sum = 0;
#pragma unroll
		for(int i = 0; i < n; ++i) sum += a[i * dim + j];
		b[j] = sum;
	}
}

__global__ void KReduceSumBlock(unsigned* a, std::size_t n, std::size_t dim, unsigned* b) {
	int i = (BSIZE / dim) * (threadIdx.x + blockIdx.x * blockDim.x) + threadIdx.y / dim;
	__shared__ float d_sum[BSIZE][BSIZE];
	if(i < n) {
		int j = threadIdx.y % dim;
		d_sum[threadIdx.x][threadIdx.y] = a[i * dim + j];
	}
	int p = threadIdx.x, q = threadIdx.y;
	__syncthreads();
	int t = BSIZE;
	t >>= 1;
	if(p < t) d_sum[p][q] += d_sum[p + t][q];
	__syncthreads();
	t >>= 1;
	if(p < t) d_sum[p][q] += d_sum[p + t][q];
	__syncthreads();
	if(threadIdx.x == 0) {
		unsigned sum = 0;
		int j = threadIdx.y;
		for(int i = 0; i < t; ++i) sum += d_sum[i][j];
		atomicAdd(b + j % dim, sum);
	}
}

void ReduceSum(unsigned* a, std::size_t n, std::size_t dim, unsigned* b) {
	if(dim >= 128) {
		int threads = BSIZE * BSIZE;
		KReduceSumLoop<<<(n + threads - 1) / threads, threads>>>(a, n, dim, b);
	} else {
		assert(BSIZE % dim == 0);
		assert(cudaMemset(b, 0, dim * sizeof(unsigned)) == cudaSuccess);
		KReduceSumBlock<<<(n * dim + BSIZE * BSIZE - 1) / (BSIZE * BSIZE), dim3(BSIZE, BSIZE)>>>(a, n, dim, b);
	}
}
