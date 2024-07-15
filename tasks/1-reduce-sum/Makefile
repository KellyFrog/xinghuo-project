CC = nvcc
FLAG = -O2

sum-v1.o: sum.cuh sum-v1.cu
	$(CC) $(FLAG) -c sum-v1.cu
sum-v2.o: sum.cuh sum-v2.cu
	$(CC) $(FLAG) -c sum-v2.cu
main.o: sum.cuh main.cu
	$(CC) $(FLAG) -c main.cu

sum-v1: main.o sum-v1.o
	$(CC) $(FLAG) -o sum-v1 main.o sum-v1.o

sum-v2: main.o sum-v2.o
	$(CC) $(FLAG) -o sum-v2 main.o sum-v2.o

.PHONY: clean
clean:
	rm main *.o

