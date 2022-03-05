// parallel.c
// mpicc parallel.c --openmp -o parallel.out

#include <mpi.h>
#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include "serial.h"

#define MASTER 0
#define MESSAGE_TAG 6969

// argv[1]: file path; argv[2]: 
int main(int argc, char* argv[]) {
	int kernel_row, kernel_col, target_row, target_col, num_targets;
	int world_size, world_rank, num_threads, pnum_targets;
	FILE *file;
	Matrix kernel;
	Matrix* arr_mat;

	// Start clock (before sscanf)
	clock_t begin = clock();
	// Thread count from args
	sscanf(argv[2], "%d", &num_threads);

	MPI_Init(NULL, NULL);

	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	if (world_rank == MASTER) {
		file = fopen(argv[1], "r");
		if (!file) {
			printf("File read error\n");
			MPI_Finalize();
			return 0;
		}

		fscanf(file, "%d %d", &kernel_row, &kernel_col);
		kernel = input_matrix_file(kernel_row, kernel_col, file);

		fscanf(file, "%d %d %d", &num_targets, &target_row, &target_col);
	}

	MPI_Bcast(&kernel_row, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(&kernel_col, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(&target_row, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(&target_col, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(&num_targets, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

	if (world_rank != MASTER) {
		init_matrix(&kernel, kernel_row, kernel_col); // Initialize matrix on other processes
	}

	for (int i = 0; i < kernel_row; i++) {
		MPI_Bcast(&(kernel.mat[i]), kernel_col, MPI_INT, MASTER, MPI_COMM_WORLD);
	}

	int size_per_process = num_targets / world_size; // size malloc per process
	int remainder = num_targets % world_size; // remainder
	pnum_targets = size_per_process;

	if ((remainder - world_rank) > 0) { // add 1 for as evenly distributed as possible
		pnum_targets++;
	}

	arr_mat = (Matrix*)malloc(pnum_targets * sizeof(Matrix));

	if (world_rank == MASTER) {
		for (int iter = 0; iter < num_targets; iter++) {
			int move_to = iter % world_size;
			Matrix move_matrix = input_matrix_file(target_row, target_col, file);
			if (move_to == MASTER) {
				arr_mat[iter / world_size] = move_matrix;
			} else {
				for (int i = 0; i < target_row; i++) {
					MPI_Send(&(move_matrix.mat[i]), target_col, MPI_INT, move_to, MESSAGE_TAG, MPI_COMM_WORLD);
				}
			}
		}
	} else {
		for (int iter = 0; iter < pnum_targets; iter++) {
			init_matrix(&(arr_mat[iter]), target_row, target_col);
			for (int i = 0; i < target_row; i++) {
				MPI_Recv(&(arr_mat[iter].mat[i]), target_col, MPI_INT, MASTER, MESSAGE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}
	}

	int arr_range[pnum_targets];

	#pragma omp parallel num_threads(num_threads)
	{
		int nthreads, tid;

		nthreads = omp_get_num_threads();
		tid = omp_get_thread_num();

		for (int iter = tid; iter < pnum_targets; iter += nthreads) {
			arr_mat[iter] = convolution(&kernel, &arr_mat[iter]);
			arr_range[iter] = get_matrix_datarange(&arr_mat[iter]);
		}
	}

	if (world_rank == MASTER) {
		int min, min_idx, master_idx = 0, idx = 0;
		int all_range[num_targets], targets_size[world_size], indices[world_size];

		merge_sort(arr_range, 0, pnum_targets - 1);

		for (int iter = 1; iter < world_size; iter++) {
			MPI_Recv(&(targets_size[iter]), 1, MPI_INT, iter, MESSAGE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(&(indices[iter]), 1, MPI_INT, iter, MESSAGE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		do {
			if (master_idx < pnum_targets) {
				min = arr_range[master_idx];
				min_idx = MASTER;
			} else {
				min_idx = 1;
				while (targets_size[min_idx] == 0) min_idx++;
				min = indices[min_idx];
			}
			for (int i = 1; i < world_size; i++) {
				if (targets_size[i] > 0 && indices[i] < min) {
					min = indices[i];
					min_idx = i;
				}
			}
			all_range[idx++] = min;
			if (min_idx == MASTER) {
				master_idx++;
			} else {
				targets_size[min_idx]--;
				if (targets_size[min_idx] > 0) {
					MPI_Recv(&(indices[min_idx]), 1, MPI_INT, min_idx, MESSAGE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
			}
		} while (idx < num_targets);

		int median = get_median(all_range, num_targets);
		int floored_mean = get_floored_mean(all_range, num_targets);
		printf("min:%d\nmax:%d\nmedian:%d\nmean:%d\n", 
			all_range[0], 
			all_range[num_targets - 1], 
			median, 
			floored_mean);
		// Time Spent
		clock_t end = clock();
		double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
		printf("time spent: %fs\n", time_spent);
	} else {
		// merge sort arr_range untuk tiap proses sebelum dilakukan MPI_Send
		// (sehingga dikirim hasil paling kecil terlebih dahulu)
		merge_sort(arr_range, 0, pnum_targets - 1);
		MPI_Send(&pnum_targets, 1, MPI_INT, MASTER, MESSAGE_TAG, MPI_COMM_WORLD);
		for (int i = 0; i < pnum_targets; i++) {
			MPI_Send(&(arr_range[i]), 1, MPI_INT, MASTER, MESSAGE_TAG, MPI_COMM_WORLD);
		}
	}

	MPI_Finalize();

	return 0;
}