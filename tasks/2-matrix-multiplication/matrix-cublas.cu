#include "matrix.cuh"
#include "cublas_v2.h"
#include "cuda_runtime.h"

void MatrixMul(const float* a, const float* b, std::size_t n, std::size_t m, std::size_t k, float* c) {
	cublasHandle_t handle;
	cublasCreate(&handle);
	float alpha = 1, beta = 0;
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, n, m, &alpha, b, k, a, m, &beta, c, k);
}
