#ifndef SERIAL_H
#define SERIAL_H

#define NMAX 100
#define DATAMAX 1000
#define DATAMIN -1000

typedef struct Matrix {
	int mat[NMAX][NMAX];	// Matrix cells
	int row_eff;			// Matrix effective row
	int col_eff;			// Matrix effective column
} Matrix;

void init_matrix(Matrix *m, int nrow, int ncol);
Matrix input_matrix(int nrow, int ncol);
Matrix input_matrix_file(int nrow, int ncol, FILE *file);
void print_matrix(Matrix *m);
int get_matrix_datarange(Matrix *m);
int supression_op(Matrix *kernel, Matrix *target, int row, int col);
Matrix convolution(Matrix *kernel, Matrix *target);
void merge_array(int *n, int left, int mid, int right);
void merge_sort(int *n, int left, int right);
void print_array(int *n, int size);
int get_median(int *n, int length);
long get_floored_mean(int *n, int length);

#endif