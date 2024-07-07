CC = nvcc
FLAG = -O2

main: main.o bubble.o
	$(CC) $(FLAG) -o main main.o bubble.o

main.o: bubble.cuh main.cu
	$(CC) $(FLAG) -c main.o main.cu

bubble.o: bubble.cuh bubble.cu
	$(CC) $(FLAG) -c bubble.o bubble.cu

.PHONY: clean
clean:
	rm main *.o
