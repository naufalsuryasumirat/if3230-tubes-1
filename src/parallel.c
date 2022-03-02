// parallel.c
// mpicc parallel.c --openmp -o parallel.out

#include <mpi.h>
#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <serial.h>
#include <stdlib.h>

// argv[1]: file path; argv[2]: 
int main(int argc, char* argv[]) {
	int kernel_row, kernel_col, target_row, target_col, num_targets;
	int world_size, world_rank, num_threads;
	
	clock_t begin = clock();

	file = fopen(argv[1], "r");
	if (!file) {
		printf("File read error\n");
		return 0;
	}

	sscanf(argv[2], "%d", &num_threads /* banyaknya thread */);

	fscanf(file, "%d %d", &kernel_row, &kernel_col);
	Matrix kernel = input_matrix(kernel_row, kernel_col);

	fscanf(file, "%d %d %d", &num_targets, &target_row, &target_col);
	Matrix* arr_mat = (Matrix*)malloc(num_targets * sizeof(Matrix));
	int arr_range[num_targets];

	MPI_Init(NULL, NULL);

	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	#pragma omp parallel num_threads(num_threads)
	{
		int nthreads, tid;
		nthreads = omp_get_num_threads();
		tid = omp_get_thread_num();
		printf("Hello world from rank %d out of %d processors, 
			from thread %d out of %d threads, %s\n", 
			world_rank, world_size, tid, nthreads);
	}

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	MPI_Finalize();

	return 0;
}