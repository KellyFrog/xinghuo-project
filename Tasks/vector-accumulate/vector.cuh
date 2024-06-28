#ifndef _CUDA_ND_VECTOR_ADD_H_
#define _CUDA_ND_VECTOR_ADD_H_

#include <cstddef>

__global__ void Accumulate(float*, std::size_t, std::size_t, float*);

#endif
