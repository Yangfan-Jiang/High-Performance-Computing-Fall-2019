#include<iostream>
#include<cstdlib>
#include<mpi.h>
#include<cmath>
#include"omp.h"

#define N 512

using namespace std;

void init_matrix(double **A, double **B, double **C, int local_i, int local_j, int block_size);
void init_matrix_s(double **A, double **B, double **C);
void matrix_multi_serial(double **A, double **B, double **C);

int main()
{
    int my_rank, comm_sz;
    int source;
    double s_time, parallel_time;
    double start, end, p_start, p_end;

    /* allocate contiguous memory */
    double *A_storage = new double[N*N];
    double *B_storage = new double[N*N];
    double *C_storage = new double[N*N];
    
    double **A = new double *[N];
    double **B = new double *[N];
    double **C = new double *[N];
    
    for(int i=0; i<N; i++) {
        A[i] = &A_storage[i*N];
        B[i] = &B_storage[i*N];
        C[i] = &C_storage[i*N];
    }

    MPI_Request request[50500];
    MPI_Request tmp;
    MPI_Status status;

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
    
    /* allocate contiguous memory */
    double *local_A_storage=new double[block_size*block_size];
    double *local_B_storage=new double[block_size*block_size];
    double *local_C_storage=new double[block_size*block_size];

    double **local_A = new double *[block_size];
    double **local_B = new double *[block_size];
    double **local_C = new double *[block_size];
    
    for(int i=0; i<block_size; i++) {
        local_A[i] = &(local_A_storage[i*block_size]);
        local_B[i] = &(local_B_storage[i*block_size]);
        local_C[i] = &(local_C_storage[i*block_size]);
    }
    
    double local_A_cal[sqrtProcNum][block_size][block_size];
    double local_B_cal[sqrtProcNum][block_size][block_size];
    
    /* 
       root process run serial version and boardcast 
       the correct result of matrix C to other processes
    */
    MPI_Barrier(MPI_COMM_WORLD);
    if(my_rank == 0) {
        init_matrix_s(A, B, C);
        start = MPI_Wtime();
        matrix_multi_serial(A, B, C);
        end = MPI_Wtime();
        s_time = end - start;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&(C[0][0]), N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    /* initialize local matrix data */
    init_matrix(local_A, local_B, local_C, local_i, local_j, block_size);
    
    /* send local A matrix to processes on same row */
    MPI_Barrier(MPI_COMM_WORLD);
    p_start = MPI_Wtime();
    
    for(int i=0; i<sqrtProcNum; i++) {
        MPI_Isend(&(local_A[0][0]), block_size*block_size, MPI_DOUBLE, rank_i*sqrtProcNum + i, my_rank, MPI_COMM_WORLD, &tmp);
    }
    
    for(int i=0; i<sqrtProcNum; i++) {
        MPI_Irecv(&(local_A_cal[i][0][0]), block_size*block_size, MPI_DOUBLE, rank_i*sqrtProcNum + i, 
                    rank_i*sqrtProcNum + i, MPI_COMM_WORLD, &request[rank_i*sqrtProcNum+i]);
    }
    
    for(int i=0; i<sqrtProcNum; i++) {
        MPI_Wait(&request[rank_i*sqrtProcNum+i], &status);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    /* send local B matrix to processes on same column */
    for(int i=0; i<sqrtProcNum; i++) {
        MPI_Isend(&(local_B[0][0]), block_size*block_size, MPI_DOUBLE, i*sqrtProcNum + rank_j, my_rank, MPI_COMM_WORLD, &tmp);
    }
    
    for(int i=0; i<sqrtProcNum; i++) {
        MPI_Irecv(&(local_B_cal[i][0][0]), block_size*block_size, MPI_DOUBLE, i*sqrtProcNum + rank_j, 
                    i*sqrtProcNum + rank_j, MPI_COMM_WORLD, &request[i*sqrtProcNum + rank_j]);
    }

    for(int i=0; i<sqrtProcNum; i++) {
        MPI_Wait(&request[i*sqrtProcNum + rank_j], &status);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    /* for each process, calculate local results */
    for(int k_mat = 0; k_mat<sqrtProcNum; k_mat++) {
        for(int i=0; i<block_size; i++)
            for(int j=0; j<block_size; j++)
                for(int k=0; k<block_size; k++) {
                    local_C[i][j] += local_A_cal[k_mat][i][k]*local_B_cal[k_mat][k][j];
                }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    p_end = MPI_Wtime();
    /* verify result */
    for(int i=0; i<block_size; i++) {
        for(int j=0; j<block_size; j++) {
            if(fabs(local_C[i][j]-C[i+local_i][j+local_j]) > 0.0001) {
                cout << my_rank << endl;
                cout << "Incorrect answer!" << endl;
                MPI_Abort(MPI_COMM_WORLD, 911);
            }
        }
    }
    
    MPI_Finalize();
    if(my_rank == 0) {
        parallel_time = p_end - p_start;
        cout << "\nVerify successfully!" << endl;
        cout << "parallel elapsed:" << parallel_time << "s" << endl;
        cout << "serial   elapsed:" << s_time << "s" << endl;
        cout << " --------- " << endl;
        cout << "size: " << N << endl;
        cout << "num of proc " << comm_sz << endl;
        cout << "Tp: " << parallel_time << endl;
        cout << "speed up: " << s_time*1.0/parallel_time << endl;
        cout << "efficiency: " << s_time*1.0/parallel_time/comm_sz << endl << endl;;
    }
    
    return 0;
}

void init_matrix(double **A, double **B, double **C, int local_i, int local_j, int block_size) {
    for(int i=0; i<block_size; i++)
        for(int j=0; j<block_size; j++) {
            int ii = i + local_i;
            int jj = j + local_j;
            A[i][j] = (ii-0.1*jj+1)*1.0/(ii+jj+1);
            B[i][j] = (jj-0.2*ii+1)*1.0/(ii*ii+jj*jj+1);
            C[i][j] = 0;
        }
}

void init_matrix_s(double **A, double **B, double **C) {
#   pragma omp parallel for
    for(int i=0; i<N; i++)
        for(int j=0; j<N; j++) {
            A[i][j] = (i-0.1*j+1)*1.0/(i+j+1);
            B[i][j] = (j-0.2*i+1)*1.0/(i*i+j*j+1);
            C[i][j] = 0;
        }
}

void matrix_multi_serial(double **A, double **B, double **C) {
    for(int i=0; i<N; i++) 
        for(int j=0; j<N; j++)
            for(int k=0; k<N; k++) {
                C[i][j] += A[i][k]*B[k][j];
            }
}