CC = nvcc
FLAG = -O2

main: main.o matrix.o
	$(CC) $(FLAG) -o main main.o matrix.o

main.o: matrix.cuh main.cu
	$(CC) $(FLAG) -c main.o main.cu

matrix.o: matrix.cuh matrix.cu
	$(CC) $(FLAG) -c matrix.o matrix.cu
