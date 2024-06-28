#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <cstddef>

const int BSIZE = 16;

__global__ void MatrixMul(const float*, const float*, std::size_t, std::size_t, std::size_t, float*);

#endif //_MATRIX_H_
