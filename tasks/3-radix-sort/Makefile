CC = nvcc
FLAG = -O2

main: main.o sort.o
	$(CC) $(FLAG) -o main main.o sort.o

main.o: sort.cuh main.cu
	$(CC) $(FLAG) -c main.o main.cu

sort.o: sort.cuh sort.cu
	$(CC) $(FLAG) -c sort.o sort.cu

.PHONY: clean
clean:
	rm main *.o
