CC = nvcc
FLAG = -O2

sum-v1.o: sum.cuh sum-v1.cu
	$(CC) $(FLAG) -c sum-v1.cu
sum-v2.o: sum.cuh sum-v2.cu
	$(CC) $(FLAG) -c sum-v2.cu
sum-v3.o: sum.cuh sum-v3.cu
	$(CC) $(FLAG) -c sum-v3.cu
main.o: sum.cuh main.cu
	$(CC) $(FLAG) -c main.cu

sum-v1: main.o sum-v1.o
	$(CC) $(FLAG) -o sum-v1 main.o sum-v1.o
sum-v2: main.o sum-v2.o
	$(CC) $(FLAG) -o sum-v2 main.o sum-v2.o
sum-v3: main.o sum-v3.o
	$(CC) $(FLAG) -o sum-v3 main.o sum-v3.o

.PHONY: clean
clean:
	rm sum-v1 sum-v2 sum-v3 *.o
