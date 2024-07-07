#include "sort.cuh"
#include <cstring>
#include <cassert>
#include <algorithm>
#include <cstdio>

const int BSIZE = 32;
const int THREADS = BSIZE * BSIZE;
const int B = 4;
const int W = 1 << B;

__global__ void MakeCnt(unsigned* a, std::size_t n, std::size_t size, int shift, int* cnt) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int l = id * size, r = (id + 1) * size;
	if(r > n) r = n;
	/*
	int cur[W];
	std::memset(cur, 0, sizeof cur);
	*/
	int mask = W - 1;
	for(int i = l; i < r; ++i) {
		++cnt[(id << B) | (a[i] >> shift & mask)];
		//++cur[a[i] >> shift & mask];
	}
	//std::memcpy(cnt + (id << B), cur, W * sizeof(int));
}

__global__ void MakePrefixSum(std::size_t n, int* cnt, int* prefix) {
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	//int id = threadIdx.x + threadIdx.y * BSIZE;
	for(int i = j + W; i < (THREADS << B); i += W) cnt[i] += cnt[i - W];
	prefix[j] = cnt[(THREADS - 1) << B | j];
	__syncthreads();
	if(j == 0) {
		for(int i = 1; i < W; ++i) prefix[i] += prefix[i-1];
		for(int i = W - 1; i > 0; --i) prefix[i] = prefix[i-1];
		prefix[0] = 0;
	}
}

__global__ void Reorder(unsigned* a, std::size_t n, std::size_t size, int shift, int* cnt, int* prefix, unsigned* b) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int l = id * size, r = (id + 1) * size;
	if(r > n) r = n;
	/*
	int cur[W], pre[W];
	std::memcpy(cur, cnt + (id << B), W * sizeof(int));
	std::memcpy(pre, prefix, W * sizeof(int));
	*/
	int mask = W - 1;
	for(int i = r - 1; i >= l; --i) {
		int x = a[i] >> shift & mask;
		//b[pre[x] + --cur[x]] = a[i];
		b[prefix[x] + --cnt[id << B | x]] = a[i];
	}
}

__global__ void Print(unsigned* a, std::size_t n) {
	for(int i = 0; i < n; ++i) printf("%u ", a[i]);
	printf("\n");
}

void RadixSort(unsigned* a, std::size_t n) {
	assert((32 + B - 1) / B % 2 == 0);
	int* cnt = NULL, *prefix = NULL;
	unsigned* b = NULL;
	std::size_t size = (n + THREADS - 1) / THREADS;
	assert(cudaMalloc((void**)&cnt, (THREADS << B) * sizeof(int)) == cudaSuccess);
	assert(cudaMalloc((void**)&prefix, W * sizeof(int)) == cudaSuccess);
	assert(cudaMalloc((void**)&b, n * sizeof(unsigned)) == cudaSuccess);
	unsigned* b0 = b;
	for(int shift = 0; shift < 32; shift += B) {
		assert(cudaMemset(cnt, 0, (THREADS << B) * sizeof(int)) == cudaSuccess);
		MakeCnt<<<BSIZE, BSIZE>>>(a, n, size, shift, cnt);
		MakePrefixSum<<<(1 << (B / 2)), (1 << (B / 2))>>>(n, cnt, prefix);
		Reorder<<<BSIZE, BSIZE>>>(a, n, size, shift, cnt, prefix, b);
		//Print<<<1, 1>>>(b, n);
		std::swap(a, b);
	}
	assert(cudaFree(cnt) == cudaSuccess);
	assert(cudaFree(prefix) == cudaSuccess);
	assert(cudaFree(b0) == cudaSuccess);
}
