#include<iostream>
#include<cstdlib>
#include<mpi.h>
#include<cmath>
#include"omp.h"

#define N 1024

using namespace std;

void init_matrix(double **A, double **B, double **C, int local_i, int local_j, int block_size);
void init_matrix_s(double **A, double **B, double **C);
void matrix_multi_serial(double **A, double **B, double **C);

int main()
{
    int my_rank, comm_sz;
    int source;
    int ndims = 2;
    double s_time, parallel_time;
    double start, end, p_start, p_end;
    
    MPI_Comm comm_cart;
    int dims[2], periods[2];
    int coords[2]={0};
    periods[0] = 1;
    periods[1] = 1;

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

    int sqrtProcNum = int(sqrt(comm_sz));
    int block_size = N / sqrtProcNum;
    int rank_i = my_rank / sqrtProcNum;
    int rank_j = my_rank % sqrtProcNum;
    int local_i = rank_i * block_size;
    int local_j = rank_j * block_size;
    
    // each dimension size of cart comm
    dims[0] = sqrtProcNum;
    dims[1] = sqrtProcNum;
    
    int rowScr, rowDest, colSrc, colDest, my_cartrank;
    
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm_cart);
    MPI_Comm_rank(comm_cart, &my_cartrank);
    MPI_Cart_coords(comm_cart, my_cartrank, ndims, coords);
    
    /* allocate contiguous memory */
    double *local_A_storage=new double[block_size*block_size];
    double *local_B_storage=new double[block_size*block_size];
    double *local_C_storage=new double[block_size*block_size];
    double *tmp_local_A_storage=new double[block_size*block_size];
    double *tmp_local_B_storage=new double[block_size*block_size];

    double **local_A = new double *[block_size];
    double **local_B = new double *[block_size];
    double **local_C = new double *[block_size];
    double **tmp_local_A = new double *[block_size];
    double **tmp_local_B = new double *[block_size];
    
    for(int i=0; i<block_size; i++) {
        local_A[i] = &(local_A_storage[i*block_size]);
        local_B[i] = &(local_B_storage[i*block_size]);
        local_C[i] = &(local_C_storage[i*block_size]);
        tmp_local_A[i] = &(tmp_local_A_storage[i*block_size]);
        tmp_local_B[i] = &(tmp_local_B_storage[i*block_size]);
    }

    /* initialize local matrix data */
    init_matrix(local_A, local_B, local_C, local_i, local_j, block_size);
    
    for(int i=0; i<block_size; i++) {
        for(int j=0; j<block_size; j++) {
            tmp_local_A[i][j] = local_A[i][j];
            tmp_local_B[i][j] = local_B[i][j];
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    p_start = MPI_Wtime();
    
    if(coords[0]>0) {
        MPI_Cart_shift(comm_cart, 1, -coords[0], &rowScr, &rowDest);
        MPI_Sendrecv(&(tmp_local_A[0][0]), block_size*block_size, MPI_DOUBLE, rowDest, coords[0],
                     &(local_A[0][0]), block_size*block_size, MPI_DOUBLE, rowScr, coords[0], MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    if(coords[1]>0) {
        MPI_Cart_shift(comm_cart, 0, -coords[1], &colSrc, &colDest);
        MPI_Sendrecv(&(tmp_local_B[0][0]), block_size*block_size, MPI_DOUBLE, colDest, coords[1],
                     &(local_B[0][0]), block_size*block_size, MPI_DOUBLE, colSrc, coords[1], MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    for(int i=0; i<block_size; i++) {
        for(int j=0; j<block_size; j++) {
            tmp_local_A[i][j] = local_A[i][j];
            tmp_local_B[i][j] = local_B[i][j];
        }
    }
    
    
    /* local calculate */
    for(int i=0; i<block_size; i++) {
        for(int j=0; j<block_size; j++) {
            for(int k=0; k<block_size; k++) {
                local_C[i][j] += local_A[i][k] * local_B[k][j];
            }
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    // cart shift sqrtProcNum-1 times
    for(int iter=1; iter<sqrtProcNum; iter++) {
        MPI_Cart_shift(comm_cart, 1, -iter, &rowScr, &rowDest);
        MPI_Sendrecv(&(tmp_local_A[0][0]), block_size*block_size, MPI_DOUBLE, rowDest, iter,
                     &(local_A[0][0]), block_size*block_size, MPI_DOUBLE, rowScr, iter, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                     
        MPI_Cart_shift(comm_cart, 0, -iter, &colSrc, &colDest);
        MPI_Sendrecv(&(tmp_local_B[0][0]), block_size*block_size, MPI_DOUBLE, colDest, iter,
                     &(local_B[0][0]), block_size*block_size, MPI_DOUBLE, colSrc, iter, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                     
        MPI_Barrier(MPI_COMM_WORLD);
        for(int i=0; i<block_size; i++) {
            for(int j=0; j<block_size; j++) {
                for(int k=0; k<block_size; k++) {
                    local_C[i][j] += local_A[i][k] * local_B[k][j];
                }
            }
        }
        
        for(int i=0; i<block_size; i++) {
            for(int j=0; j<block_size; j++) {
                tmp_local_A[i][j] = local_A[i][j];
                tmp_local_B[i][j] = local_B[i][j];
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    p_end = MPI_Wtime();

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
    
    MPI_Finalize();

    
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