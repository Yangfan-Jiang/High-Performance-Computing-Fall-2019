#include<iostream>
#include<cstdlib>
#include<mpi.h>
#include<cmath>
#include"omp.h"

#define N 300

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
    
    int procNumPerDim = int(pow(comm_sz+1, 1.0/3));
    int block_size = N / procNumPerDim;
    
    // each dimension size of cart comm
    dims[0] = procNumPerDim;
    dims[1] = procNumPerDim;
    dims[2] = procNumPerDim;
    
    // creat cart comm
    int my_cartrank;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &comm_cart);
    MPI_Comm_rank(comm_cart, &my_cartrank);
    MPI_Cart_coords(comm_cart, my_cartrank, ndims, coords);
    
    int local_i = coords[1]*block_size;
    int local_j = coords[2]*block_size;
    
    /* allocate contiguous memory */
    double *local_A_storage = new double[block_size*block_size];
    double *local_B_storage = new double[block_size*block_size];
    double *local_C_storage = new double[block_size*block_size];
    double *result_C_storage = new double[block_size*block_size];

    double **local_A = new double *[block_size];
    double **local_B = new double *[block_size];
    double **local_C = new double *[block_size];
    double **result_C = new double *[block_size];
    
    for(int i=0; i<block_size; i++) {
        local_A[i] = &(local_A_storage[i*block_size]);
        local_B[i] = &(local_B_storage[i*block_size]);
        local_C[i] = &(local_C_storage[i*block_size]);
        result_C[i] = &(result_C_storage[i*block_size]);
    }
    
    MPI_Comm comm_cart_rows, comm_cart_cols;
    // split mpi comm by row, col number as numproc
    int remain_dims[3] = {0, 0, 1};
    MPI_Cart_sub(comm_cart, remain_dims, &comm_cart_rows);
    
    // split mpi comm by col
    remain_dims[1] = 1; remain_dims[2] = 0;
    MPI_Cart_sub(comm_cart, remain_dims, &comm_cart_cols);
    
    MPI_Barrier(MPI_COMM_WORLD);
    p_start = MPI_Wtime();
    // initialize matrix A and B
    if(coords[0] == 0) {
        init_matrix(local_A, local_B, local_C, result_C, local_i, local_j, block_size);
        
        int dest;
        // send partition matrix A to each processes
        int dest_coords[3] = {coords[2], coords[1], coords[2]};
        MPI_Cart_rank(comm_cart, dest_coords, &dest);
        MPI_Send(&(local_A[0][0]), block_size*block_size, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
        
        // send partition matrix B to each processes
        dest_coords[0] = coords[1];
        MPI_Cart_rank(comm_cart, dest_coords, &dest);
        MPI_Send(&(local_B[0][0]), block_size*block_size, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
    } else {
        int source;
        int source_coords[3] = {0, coords[1], coords[2]};
        MPI_Cart_rank(comm_cart, source_coords, &source);
        // recv data of matrix A
        if(coords[2] == coords[0]) {
            MPI_Recv(&(local_A[0][0]), block_size*block_size, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        // recv data of matrix B
        if(coords[1] == coords[0]) {
            MPI_Recv(&(local_B[0][0]), block_size*block_size, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
    }
    MPI_Bcast(&(local_A[0][0]), block_size*block_size, MPI_DOUBLE, coords[0], comm_cart_rows);
    MPI_Bcast(&(local_B[0][0]), block_size*block_size, MPI_DOUBLE, coords[0], comm_cart_cols);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    for(int i=0; i<block_size; i++) {
        for(int j=0; j<block_size; j++) {
            for(int k=0; k<block_size; k++) {
                local_C[i][j] += local_A[i][k]*local_B[k][j];
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    // reduction the result
    MPI_Comm comm_cart_layer;
    remain_dims[0] = 1; remain_dims[1] = 0; remain_dims[2] = 0;
    MPI_Cart_sub(comm_cart, remain_dims, &comm_cart_layer);
    MPI_Reduce(&(local_C[0][0]), &(result_C[0][0]), block_size*block_size, MPI_DOUBLE, MPI_SUM, 0, comm_cart_layer);
    MPI_Barrier(MPI_COMM_WORLD);
    p_end = MPI_Wtime();
    
    init_matrix_s(A, B, C);
    
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
    if(coords[0] == 0) {
        for(int i=0; i<block_size; i++) {
            for(int j=0; j<block_size; j++) {
                if(fabs(result_C[i][j]-C[i+local_i][j+local_j]) > 0.0001) {
                    cout << "Incorrect answer!" << endl;
                    MPI_Abort(MPI_COMM_WORLD, 911);
                }
            }
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

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
    for(int i=0; i<N; i++) 
        for(int j=0; j<N; j++)
            for(int k=0; k<N; k++) {
                C[i][j] += A[i][k]*B[k][j];
            }
}