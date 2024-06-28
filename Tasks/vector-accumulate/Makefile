CC = nvcc
FLAG = -O2

main: main.o vector.o
	$(CC) -o main main.o vector.o $(FLAG)
vector.o: vector.cuh vector.cu
	$(CC) -c vector.o vector.cu $(FLAG)
main.o: vector.cuh main.cu
	$(CC) -c main.o main.cu $(FLAG)

.PHONY: clean
clean:
	rm main *.o
