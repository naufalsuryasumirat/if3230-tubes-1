// parallel.c
// mpicc parallel.c --openmp -o parallel.out

#include <mpi.h>
#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
	MPI_Init(NULL, NULL);

	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);


	#pragma omp parallel num_threads(8)
	{
		int nthreads, tid;
		nthreads = omp_get_num_threads();
		tid = omp_get_thread_num();
		printf("Hello world from processor %s, rank %d out of %d processors, from thread %d out of %d threads\n", processor_name, world_rank, world_size, tid, nthreads);
	}

	MPI_Finalize();
}