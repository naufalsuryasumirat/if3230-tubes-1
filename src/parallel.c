// parallel.c
// mpicc parallel.c --openmp -o parallel.out

#include <mpi.h>
#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include "serial.h"

#define ROOT 0

// argv[1]: file path; argv[2]: 
int main(int argc, char* argv[]) {
	int kernel_row, kernel_col, target_row, target_col, num_targets;
	int world_size, world_rank, num_threads, pnum_targets;
	FILE *file;
	Matrix kernel;
	Matrix* arr_mat;
	
	clock_t begin = clock();
	// Thread count from args
	sscanf(argv[2], "%d", &num_threads);

	MPI_Init(NULL, NULL);

	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	if (world_rank == ROOT) {
		file = fopen(argv[1], "r");
		if (!file) {
			printf("File read error\n");
			MPI_Finalize();
			return 0;
		}

		fscanf(file, "%d %d", &kernel_row, &kernel_col);
		kernel = input_matrix(kernel_row, kernel_col, file);

		fscanf(file, "%d %d %d", &num_targets, &target_row, &target_col);
	}

	MPI_Bcast(&kernel_row, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
	MPI_Bcast(&kernel_col, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
	MPI_Bcast(&target_row, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
	MPI_Bcast(&target_col, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
	MPI_Bcast(&num_targets, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

	if (world_rank != ROOT) { // init kernel matrix
		init_matrix(&kernel, kernel_row, kernel_col);
	}

	for (int i = 0; i < kernel_row; i++) {
		for (int j = 0; j < kernel_col; j++) {
			MPI_Bcast(&(kernel.mat[i][j]), 1, MPI_INT, ROOT, MPI_COMM_WORLD);
		}
	}

	int size_per_process = num_targets / world_size; // size malloc per process

	if (world_rank != world_size - 1) {
		pnum_targets = size_per_process;
	} else {
		pnum_targets = num_targets % world_size;
		if (pnum_targets == 0) {
			pnum_targets = size_per_process;
		}
	}

	arr_mat = (Matrix*)malloc(pnum_targets * sizeof(Matrix));

	// distribute matrices from root to other processes
	// calculate convolution in all processes
	// gather all result from all process
	// print result

	#pragma omp parallel num_threads(num_threads)
	{
		int nthreads, tid;
		nthreads = omp_get_num_threads();
		tid = omp_get_thread_num();
		printf("Hello world from rank %d out of %d processors, from thread %d out of %d threads\n", world_rank, world_size, tid, nthreads);
		// for (int i = 0; i < num_targets; i++) {
			
		// }
		print_matrix(&kernel);
		printf("\n");
	}

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	MPI_Finalize();

	if (world_rank == ROOT) {
		printf("time spent: %fs\n", time_spent);
	}

	return 0;
}