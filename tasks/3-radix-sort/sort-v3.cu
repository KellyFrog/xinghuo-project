#include "sort.cuh"
#include <cstring>
#include <cassert>
#include <algorithm>
#include <cstdio>

const int B = 4;
const int T = 8;

typedef unsigned long long uint64_t;
typedef unsigned short uint16_t;

__global__ void KGetLowbits(unsigned* d_a, std::size_t n, int shift, int* d_cnt, int* d_suf) {
	//16 * 4 = 64
	__shared__ uint64_t cnt[1 << T][2], tmp[1 << T][2];
	__shared__ uint16_t app[1 << T], a[1 << T];
	__shared__ int sum;
	int idx = threadIdx.x;
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	cnt[idx][0] = cnt[idx][1] = 0;
	app[idx] = 0;
	sum = 0;
	if(i < n) {
		int lowbits = d_a[i] >> shift & ((1 << B) - 1);
		a[idx] = lowbits;
		cnt[idx][lowbits >> 3] = 1ull << ((lowbits & 7) << 3);
		app[idx] = 1 << lowbits;
		//printf("[%d] lowbit = %d\n", idx, lowbits);
	}
	__syncthreads();
	int t = 1 << T;

	while(t > 1) {
		t >>= 1;
		if(t + idx < (1 << T)) {
			tmp[idx][0] = cnt[idx][0] + cnt[idx + t][0];
			tmp[idx][1] = cnt[idx][1] + cnt[idx + t][1];
			app[idx] |= app[idx + t];
		} else {
			tmp[idx][0] = cnt[idx][0];
			tmp[idx][1] = cnt[idx][1];
		}
		__syncthreads();
		cnt[idx][0] = tmp[idx][0];
		cnt[idx][1] = tmp[idx][1];
		__syncthreads();
	}
	//if(i < n) printf("[%d] = %llu %llu %u\n", idx, cnt[idx][0], cnt[idx][1], (unsigned)app[idx]);
	int startpos = blockIdx.x << B;
	if(idx < (1 << B)) {
		if(app[0] >> idx & 1) {
			//printf("get %d in app\n", idx);
			int x = (cnt[0][idx >> 3] >> ((idx & 7) << 3) & ((1 << T) - 1));
			if(!x) x = 1 << T;
			d_cnt[startpos + idx] = x;
			atomicAdd(&sum, x);
		} else {
			d_cnt[startpos + idx] = 0;
		}
	}
	if(i < n) {
		int lowbits = a[idx];
		int x = cnt[idx][lowbits >> 3] >> ((lowbits & 7) << 3) & ((1 << T) - 1);
		if(!x) x = 1 << T;
		d_suf[i] = x;
		//printf("d_suf[%d] = %d %d\n", i, x, lowbits);
	}
	__syncthreads();
	/*
	if(idx == 0) {
		int mi = 1 << T;
		if(n - i < mi) mi = n - i;
		if(sum != mi) printf("%d %d\n", sum, mi);
		assert(sum == mi);
	}
	*/
	//if(d_cnt[startpos + idx]) printf("d_cnt[%d] = %d\n", startpos + idx, d_cnt[startpos + idx]);
}

__global__ void KReorder(unsigned* d_a, std::size_t n, int shift, int* d_cnt, int* d_prefix, int* d_suf, unsigned* d_b) {
	int startpos = blockIdx.x << B;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i < n) {
		int lowbits = d_a[i] >> shift & ((1 << B) - 1);
		int t = d_prefix[lowbits] + d_cnt[startpos | lowbits] - d_suf[i];
		d_b[t] = d_a[i];
		//printf("d_b[%d - %d] = d_a[%d] = %u %d\n", d_prefix[lowbits] + d_cnt[startpos | lowbits], d_suf[i], i, d_a[i], lowbits);
		//printf("d_b[%d] = d_a[%d] = %u %d\n", t, i, d_a[i], lowbits);
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

__global__ void KExclusivePrefix(int* d_cnt, int n) {

	int sum = 0;
	for(int i = 0; i < (1 << B); ++i) {
		//printf("[%d] = %d\n", i, d_cnt[i]);
		int x = d_cnt[i];
		d_cnt[i] = sum;
		sum += x;
	}
	//printf("sum = %d\n", sum);
	assert(sum == n);
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
		KMoveBackwards<<<(n + (1 << B)) / (1 << B), dim3((1 << B), (1 << B))>>>(d_cnt, n, t, d_temp);
		std::swap(d_cnt, d_temp);
	}
		//KPrint<<<1, 1>>>(d_cnt, n << B);
}

void RadixSort(unsigned* d_a, std::size_t n) {
	int *d_cnt = NULL, *d_tempCnt = NULL, *d_prefix = NULL, *d_suf = NULL;
	unsigned* d_b = NULL;
	int t = (n + (1 << T) - 1) / (1 << T);
	assert(cudaMalloc((void**)&d_cnt, (t << B) * sizeof(int)) == cudaSuccess);
	assert(cudaMalloc((void**)&d_tempCnt, (t << B) * sizeof(int)) == cudaSuccess);
	assert(cudaMalloc((void**)&d_prefix, (1 << B) * sizeof(int)) == cudaSuccess);
	assert(cudaMalloc((void**)&d_b, n * sizeof(unsigned)) == cudaSuccess);
	assert(cudaMalloc((void**)&d_suf, n * sizeof(unsigned)) == cudaSuccess);
	for(int shift = 0; shift < 32; shift += B) {
		KGetLowbits<<<t, (1 << T)>>>(d_a, n, shift, d_cnt, d_suf);
		MakePrefix(d_cnt, t, d_tempCnt);
		assert(cudaMemcpy(d_prefix, d_cnt + ((t - 1) << B), (1 << B) * sizeof(int), cudaMemcpyDeviceToDevice) == cudaSuccess);
		KExclusivePrefix<<<1, 1>>>(d_prefix, n);
		//KPrint<<<1, 1>>>(d_prefix, (1 << B));
		KReorder<<<t, (1 << T)>>>(d_a, n, shift, d_cnt, d_prefix, d_suf, d_b);
		std::swap(d_a, d_b);
	}
	assert(cudaFree(d_cnt) == cudaSuccess);
	assert(cudaFree(d_tempCnt) == cudaSuccess);
	assert(cudaFree(d_prefix) == cudaSuccess);
	assert(cudaFree(d_b) == cudaSuccess);
	assert(cudaFree(d_suf) == cudaSuccess);
}
