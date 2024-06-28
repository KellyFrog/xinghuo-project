#include "matrix.cuh"
#include <cstdio>

__global__ void MatrixMul(const float* a, const float* b, std::size_t n, std::size_t m, std::size_t k, float* c) {
	__shared__ float sa[BSIZE][BSIZE];
	__shared__ float sb[BSIZE][BSIZE];
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int x = threadIdx.x;
	int y = threadIdx.y;
	float res = 0;
	for(int p = 0; p < m; p += BSIZE) {
		if(y == 0) {
			for(int q = 0; q < BSIZE; ++q) {
				if(i < n && p + q < m) sa[x][q] = a[i * m + p + q];
				else sa[x][q] = 0;
			}
		}
		if(x == 0) {
			for(int q = 0; q < BSIZE; ++q) {
				if(p + q < m && j < k) {
					sb[q][y] = b[(p + q) * k + j];
				}
				else sb[q][y] = 0;
			}
		}
		__syncthreads();
		for(int q = 0; q < BSIZE; ++q) res += sa[x][q] * sb[q][y];
		__syncthreads();
	}
	if(i < n && j < k) c[i * k + j] = res;
}
