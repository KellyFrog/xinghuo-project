#include "bubble.cuh"

#include <random>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <cstring>

int main(int argc, char** argv) {
	assert(argc == 3);
	std::size_t n = std::atoi(argv[1]);
	std::size_t t = std::atoi(argv[2]);
	unsigned *a = (unsigned*) std::malloc(n * sizeof(unsigned));
	unsigned *b = (unsigned*) std::malloc(n * sizeof(unsigned));
	unsigned *d_a = NULL;
	std::mt19937 rnd(0x20061013);
	int k = rnd() % n + 1;
	std::cerr << "k = " << k << "\n";
	for(int i = 0; i < n; ++i) a[i] = rnd();
	for(int i = 0; i < k; ++i) a[rnd() % n] = 0;
	std::memcpy(b, a, n * sizeof(unsigned));
	assert(cudaMalloc((void**)&d_a, n * sizeof(unsigned)) == cudaSuccess);
	int res = 0, ans = 0;
	for(int i = 0; i < t; ++i) {
		assert(cudaMemcpy(d_a, a, n * sizeof(unsigned), cudaMemcpyHostToDevice) == cudaSuccess);
		res = SqeezeBubble(d_a, n);
	}
	assert(cudaMemcpy(a, d_a, n * sizeof(unsigned), cudaMemcpyDeviceToHost) == cudaSuccess);
	for(int i = 0; i < n; ++i) {
		if(b[i] != 0) {
			b[ans++] = b[i];
		}
	}
	assert(res == ans);
	for(int i = 0; i < ans; ++i) assert(a[i] == b[i]);
	std::free(a);
	std::free(b);
	assert(cudaFree(d_a) == cudaSuccess);

	return 0;
}
