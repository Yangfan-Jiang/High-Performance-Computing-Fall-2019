#include"gen_list_ranking_data.c" 
#include<iostream>
#include<mpi.h>

#define N 100000000

using namespace std;


int main()
{
    int my_rank, comm_sz;
    int source;
    double start, end;
    
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    int local_len = N/comm_sz;
    int my_start = my_rank*local_len;
    int my_end = my_start + local_len;
    
    // alltoall
    int* d = (int*)malloc(N*sizeof(int));
	int* n = NULL;
	n = gen_linked_list_2(N);

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    
    // initialize
    for(int i=0; i<N; i++) {
		d[i] = n[i]==-1?0:1;
	}
    
    int max = -1;
    int global_max = -1;
    while(global_max<N-1) {
        max = global_max;
        for(int i=my_start; i<my_end; i++) {
            if(n[i] != -1) {
				d[i] = d[i] + d[n[i]];
				n[i] = n[n[i]];
				if(max < d[i]) {
					max = d[i];
				}
			}
        }
        // communication
        // 1. get the max num
        MPI_Allreduce(&max, &global_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allgather(n+my_start, local_len, MPI_INT, n, local_len, MPI_INT, MPI_COMM_WORLD);
        MPI_Allgather(d+my_start, local_len, MPI_INT, d, local_len, MPI_INT, MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    if(my_rank == 0) {
        cout << "MPI elapsed:" << end-start << "s" << endl;
    }

    
    MPI_Finalize();
	return 0;
}
