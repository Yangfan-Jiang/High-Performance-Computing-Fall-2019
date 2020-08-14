#include<iostream>
#include<cstdlib>
#include<mpi.h>
#include<cmath>
#include"omp.h"

#define N 512

using namespace std;


double x[N];
double y[N];

void init_matrix(double **A, double *x, int local_i, int local_j, int block_size);
void init_matrix_s(double **A, double *x);
void matrix_multi(double **A);

int main()
{   
    int my_rank, comm_sz;
    int source;
    double s_time, parallel_time;
    double start, end;
    
    double **A = new double *[N];
    for(int i=0; i<N; i++)
        A[i] = new double[N];
    
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
        
    int sqrtProcNum = int(sqrt(comm_sz));
    int block_size = N / sqrtProcNum;
    int rank_i = my_rank / sqrtProcNum;
    int rank_j = my_rank % sqrtProcNum;
    int local_i = rank_i * block_size;
    int local_j = rank_j * block_size;
    
    double local_x[block_size];
    double Res[block_size];
    double **local_A = new double *[block_size];
    
    for(int i=0; i<block_size; i++)
        local_A[i] = new double[block_size];
    
    /* 
       root process run serial version and boardcast 
       the correct result of y to other processes
    */
    MPI_Barrier(MPI_COMM_WORLD);
    if(my_rank == 0) {
        init_matrix_s(A, x);
        start = MPI_Wtime();
        matrix_multi(A);
        end = MPI_Wtime();
        s_time = end - start;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(y, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    /* initialize local matrix and vector */
    init_matrix(local_A, local_x, local_i, local_j, block_size);
    
    for(int i=0; i<block_size; i++) 
        Res[i] = 0;
    
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    
    /* local matrix-vector multiply */
    for(int i=0; i<block_size; i++) {
        for(int j=0; j<block_size; j++) {
            Res[i] += local_A[i][j] * local_x[j];
        }
    }
    
    /* reduce the result */
    if (rank_j == sqrtProcNum-1) {
        double recv_res[block_size];
        for(int j=0; j<sqrtProcNum-1; j++) {
            MPI_Recv(&recv_res, block_size, MPI_DOUBLE, 
                     rank_i*sqrtProcNum + j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for(int i=0 ;i<block_size; i++) {
                Res[i] += recv_res[i];
            }
        }
    } else {
        MPI_Send(&Res, block_size, MPI_DOUBLE, 
                 rank_i*sqrtProcNum + sqrtProcNum-1, 0, MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    if (my_rank == 0) {
        parallel_time = end - start;
    }
    
    /* verify result */
    if (rank_j == sqrtProcNum-1) {
        for(int i=0; i<block_size; i++) {
            if (fabs(Res[i] - y[local_i + i]) > 0.00001) {
                cout << my_rank << endl;
                cout << "Incorrect answer!" << endl;
                MPI_Abort(MPI_COMM_WORLD, 911);
            }
        }
    }
    
    MPI_Finalize();
    if(my_rank == 0) {
        cout << "Verify successfully!" << endl;
        cout << "parallel elapsed:" << parallel_time << "s" << endl;
        cout << "serial   elapsed:" << s_time << "s" << endl;
        cout << "\n ----- \n" << endl;
        cout << "size: " << N << endl;
        cout << "num of proc " << comm_sz << endl;
        cout << "Tp: " << parallel_time << endl;
        cout << "speed up: " << s_time*1.0/parallel_time << endl;
        cout << "efficiency: " << s_time*1.0/parallel_time/comm_sz << endl;
    }
    
    return 0;
}


void init_matrix(double **A, double *x, int local_i, int local_j, int block_size) {
    for(int i=0; i<block_size; i++)
        for(int j=0; j<block_size; j++) 
            A[i][j] = (local_i+i-0.1*(local_j+j)+1)*1.0/(local_i+i+local_j+j+1);
    
    for(int i=0; i<block_size; i++) {
        x[i] = (local_j+i)*1.0/((local_j+i)*(local_j+i)+1);
    }
}

void init_matrix_s(double **A, double *x) {
#   pragma omp parallel for
    for(int i=0; i<N; i++)
        for(int j=0; j<N; j++) 
            A[i][j] = (i-0.1*j+1)*1.0/(i+j+1);
    
    for(int i=0; i<N; i++) {
        x[i] = i*1.0/(i*i+1);
        y[i] = 0;
    }
}

void matrix_multi(double **A) {
    for(int i=0; i < N; i++)
        for(int j=0; j < N; j++)
            y[i] += A[i][j] * x[j];
}

