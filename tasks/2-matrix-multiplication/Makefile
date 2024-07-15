CC = nvcc
FLAG = -O2

main.o: main.cu matrix.cuh
	$(CC) $(FLAG) -c main.cu

matrix-cublas.o: matrix-cublas.cu matrix.cuh
	$(CC) $(FLAG) -c matrix-cublas.cu

matrix-v1.o: matrix-v1.cu matrix.cuh
	$(CC) $(FLAG) -c matrix-v1.cu

matrix-cublas: matrix-cublas.o main.o
	$(CC) $(FLAG) -o matrix-cublas matrix-cublas.o main.o -lcublas
	
matrix-v1: matrix-v1.o main.o
	$(CC) $(FLAG) -o matrix-v1 matrix-v1.o main.o
	
.PHONY:
clean:
	rm *.o matrix-v1 matrix-cublas

.PHONY:
all: matrix-cublas matrix-v1 matrix-cublas
	
