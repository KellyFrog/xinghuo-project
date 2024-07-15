CC = nvcc
FLAG = -O2

main.o: main.cu sort.cuh
	$(CC) $(FLAG) -c main.cu

sort-cub.o: sort-cub.cu sort.cuh
	$(CC) $(FLAG) -c sort-cub.cu

sort-v1.o: sort-v1.cu sort.cuh
	$(CC) $(FLAG) -c sort-v1.cu

sort-v2.o: sort-v2.cu sort.cuh
	$(CC) $(FLAG) -c sort-v2.cu

sort-v3.o: sort-v3.cu sort.cuh
	$(CC) $(FLAG) -c sort-v3.cu

sort-cub: sort-cub.o main.o
	$(CC) $(FLAG) -o sort-cub sort-cub.o main.o
	
sort-v1: sort-v1.o main.o
	$(CC) $(FLAG) -o sort-v1 sort-v1.o main.o

sort-v2: sort-v2.o main.o
	$(CC) $(FLAG) -o sort-v2 sort-v2.o main.o

sort-v3: sort-v3.o main.o
	$(CC) $(FLAG) -o sort-v3 sort-v3.o main.o
	
.PHONY:
clean:
	rm *.o sort-v1 sort-v2 sort-v3 sort-cub

.PHONY:
all: sort-cub sort-v1 sort-v2 sort-v3
	
