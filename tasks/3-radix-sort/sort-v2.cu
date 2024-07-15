#include "sort.cuh"
#include <cstring>
#include <cassert>
#include <algorithm>
#include <cstdio>

const int B = 4;

typedef unsigned long long uint64_t;
typedef unsigned short uint16_t;

__global__ void KGetLowbits(unsigned* d_a, std::size_t n, int shift, int* d_cnt, int* d_suf) {
	//16 * 4 = 64
	__shared__ uint64_t cnt[1 << B];
	__shared__ uint16_t app[1 << B], a[1 << B];
	int idx = threadIdx.x;
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int mask = (1 << B) - 1;
	cnt[idx] = 0;
	app[idx] = 0;
	if(i < n) {
		int lowbits = d_a[i] >> shift & mask;
		a[idx] = lowbits;
		cnt[idx] = 1ull << (4 * lowbits);
		app[idx] = 1 << lowbits;
	}
	__syncthreads();
	int t = 16;
	while(t > 1) {
		t >>= 1;
		if(t <= idx) {
			cnt[idx - t] += cnt[idx];
			app[idx - t] |= app[idx];
		}
		__syncthreads();
	}
	//printf("[%d] = %llu %u\n", idx, cnt[idx], (unsigned)app[idx]);
	int startpos = blockDim.x * blockIdx.x;
	int x = ((cnt[0] >> (idx * 4)) & 15);
	if((app[0] & (app[0] - 1)) == 0) {
		if(app[0] >> idx & 1) {
			d_cnt[startpos + idx] = x ? x : 16;
		} else {
			d_cnt[startpos + idx] = 0;
		}
	} else {
		d_cnt[startpos + idx] = x;
	}
	int st = app[0];
	if(i < n) {
		int lowbits = a[idx];
		int c = (cnt[idx] >> (lowbits * 4) & 15);
		if((st & (st - 1)) == 0) {
			c = c ? c : 16;
		}
		d_suf[i] = c;
	}
	//if(d_cnt[startpos + idx]) printf("d_cnt[%d] = %d\n", startpos + idx, d_cnt[startpos + idx]);
}

__global__ void KReorder(unsigned* d_a, std::size_t n, int shift, int* d_cnt, int* d_prefix, int* d_suf, unsigned* d_b) {
	/*
	__shared__ uint64_t cnt[1 << B];
	__shared__ uint16_t app[1 << B], a[1 << B];
	__shared__ int prefix[1 << B];
	int idx = threadIdx.x;
	int startpos = blockDim.x * blockIdx.x;
	cnt[idx] = 0;
	app[idx] = 0;
	prefix[idx] = d_prefix[idx] + d_cnt[i];
	if(i < n) {
		a[idx] = d_a[i] >> shift & mask;
		int lowbits = a[idx];
		cnt[idx] = 1ull << (4 * lowbits);
		app[idx] = 1 << lowbits;
	}
	__syncthreads();
	int t = 16;
	while(t > 1) {
		t >>= 1;
		if(t <= idx) {
			cnt[idx - t] += cnt[idx];
			app[idx - t] |= app[idx];
		}
		__syncthreads();
	}
	int st = app[0];
	*/
	int idx = threadIdx.x;
	int startpos = blockDim.x * blockIdx.x;
	int i = startpos + idx;
	int mask = (1 << B) - 1;
	if(i < n) {
		int lowbits = d_a[i] >> shift & mask;
		int t = d_prefix[lowbits] + d_cnt[startpos | lowbits] - d_suf[i];
		d_b[t] = d_a[i];
		//printf("d_b[%d - %d] = d_a[%d] = %u %d\n", d_prefix[lowbits] + d_cnt[lowbits], d_suf[i], i, d_a[i], lowbits);
	}
}

__global__ void KMoveBackwards(int* d_src, std::size_t n, int t, int* d_dst) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y;
	if(i < n) {
		if(t <= i) {
			d_dst[i << 4 | j] = d_src[i << 4 | j] + d_src[(i - t) << 4 | j];
		} else {
			d_dst[i << 4 | j] = d_src[i << 4 | j];
		}
	}
}

__global__ void KExclusivePrefix(int* d_cnt) {
	int sum = 0;
	for(int i = 0; i < (1 << B); ++i) {
		int x = d_cnt[i];
		d_cnt[i] = sum;
		sum += x;
	}
}

__global__ void KPrint(int* a, std::size_t n) {
	/*
	for(int i = 0; i < n; ++i) {
		printf("%d ", a[i]);
	}
	printf("\n");
	*/
	printf("---\n");
	for(int i = 0; i < n; ++i) 
		if(a[i]) printf("[%d] = %d\n", i, a[i]);
}

void MakePrefix(int* &d_cnt, std::size_t n, int* &d_temp) {
	int t = 1;
	while(t < n) t <<= 1;
	while(t > 1) {
		t >>= 1;
		KMoveBackwards<<<n, dim3((1 << B), (1 << B))>>>(d_cnt, n, t, d_temp);
		std::swap(d_cnt, d_temp);
	}
		//KPrint<<<1, 1>>>(d_cnt, n << B);
}

void RadixSort(unsigned* d_a, std::size_t n) {
	int *d_cnt = NULL, *d_tempCnt = NULL, *d_prefix = NULL, *d_suf = NULL;
	unsigned* d_b = NULL;
	int t = (n + (1 << B) - 1) / (1 << B);
	assert(cudaMalloc((void**)&d_cnt, (t << B) * sizeof(int)) == cudaSuccess);
	assert(cudaMalloc((void**)&d_tempCnt, (t << B) * sizeof(int)) == cudaSuccess);
	assert(cudaMalloc((void**)&d_prefix, (1 << B) * sizeof(int)) == cudaSuccess);
	assert(cudaMalloc((void**)&d_b, n * sizeof(unsigned)) == cudaSuccess);
	assert(cudaMalloc((void**)&d_suf, n * sizeof(unsigned)) == cudaSuccess);
	for(int shift = 0; shift < 32; shift += B) {
		KGetLowbits<<<t, (1 << B)>>>(d_a, n, shift, d_cnt, d_suf);
		MakePrefix(d_cnt, t, d_tempCnt);
		assert(cudaMemcpy(d_prefix, d_cnt + ((t - 1) << B), (1 << B) * sizeof(int), cudaMemcpyDeviceToDevice) == cudaSuccess);
		KExclusivePrefix<<<1, 1>>>(d_prefix);
		//KPrint<<<1, 1>>>(d_prefix, (1 << B));
		KReorder<<<t, (1 << B)>>>(d_a, n, shift, d_cnt, d_prefix, d_suf, d_b);
		std::swap(d_a, d_b);
	}
	assert(cudaFree(d_cnt) == cudaSuccess);
	assert(cudaFree(d_tempCnt) == cudaSuccess);
	assert(cudaFree(d_prefix) == cudaSuccess);
	assert(cudaFree(d_b) == cudaSuccess);
}
