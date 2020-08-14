#include<iostream>
#include<cstdlib>
#include<mpi.h>
#include<cmath>
#include"omp.h"
#include<fstream>

#define N 6912

using namespace std;

void init_matrix(double **A, double **B, double **C, double **C2 ,int local_i, int local_j, int block_size);
void init_matrix_s(double **A, double **B, double **C);
void matrix_multi_serial(double **A, double **B, double **C);


int main() {
    int my_rank, comm_sz;
    double s_time, parallel_time;
    double start, end, p_start, p_end;
    
    int ndims = 3;
    int dims[3];
    int periods[3] = {1, 1, 1};
    int coords[3] = {0};
    MPI_Comm comm_cart;
    
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


    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
    
    MPI_Barrier(MPI_COMM_WORLD);
    p_start = MPI_Wtime();
   
    matrix_multi_serial(A, B, C);
    
    MPI_Barrier(MPI_COMM_WORLD);
    p_end = MPI_Wtime();
    
    init_matrix_s(A, B, C);
    

    MPI_Barrier(MPI_COMM_WORLD);

    if(my_rank == 0) {
        parallel_time = p_end - p_start;
        cout << "\nVerify successfully!" << endl;
        cout << "parallel elapsed:" << parallel_time << "s" << endl;
        cout << " --------- " << endl;
        cout << "size: " << N << endl;
        cout << "num of proc " << comm_sz << endl;
    }

    MPI_Finalize();

    return 0;
    
}




void init_matrix(double **A, double **B, double **C, double **res_C, int local_i, int local_j, int block_size) {
    for(int i=0; i<block_size; i++) {
        for(int j=0; j<block_size; j++) {
            int ii = i + local_i;
            int jj = j + local_j;
            
            A[i][j] = (ii-0.1*jj+1)*1.0/(ii+jj+1);
            B[i][j] = (jj-0.2*ii+1)*1.0/(ii*ii+jj*jj+1);
            C[i][j] = 0;
            res_C[i][j] = 0;
        }
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
    int i, j, k;
    for(i=0; i<N; i++) {
#       pragma omp parallel for shared(A,B,C) private(k)
        for(j=0; j<N; j++) {
            for(k=0; k<N; k++) {
                C[i][j] += A[i][k]*B[k][j];
            }
        }
    }
}
