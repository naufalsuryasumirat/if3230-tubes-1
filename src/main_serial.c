#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "serial.h"

// main() driver
int main() {
	int kernel_row, kernel_col, target_row, target_col, num_targets;
	
	clock_t begin = clock();

	// reads kernel's row and column and initalize kernel matrix from input
	scanf("%d %d", &kernel_row, &kernel_col);
	Matrix kernel = input_matrix(kernel_row, kernel_col);
	
	// reads number of target matrices and their dimensions.
	// initialize array of matrices and array of data ranges (int)
	scanf("%d %d %d", &num_targets, &target_row, &target_col);
	Matrix* arr_mat = (Matrix*)malloc(num_targets * sizeof(Matrix));
	int arr_range[num_targets];
	
	// read each target matrix, compute their convolution matrices, and compute their data ranges
	for (int i = 0; i < num_targets; i++) {
		arr_mat[i] = input_matrix(target_row, target_col);
		arr_mat[i] = convolution(&kernel, &arr_mat[i]);
		arr_range[i] = get_matrix_datarange(&arr_mat[i]); 
	}

	// sort the data range array
	merge_sort(arr_range, 0, num_targets - 1);
	
	int median = get_median(arr_range, num_targets);
	int floored_mean = get_floored_mean(arr_range, num_targets);

	// print the min, max, median, and floored mean of data range array
	printf("min:%d\nmax:%d\nmedian:%d\nmean:%d\n", 
			arr_range[0], 
			arr_range[num_targets - 1], 
			median, 
			floored_mean);
			
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("time spent: %fs\n", time_spent);
	
	return 0;
}