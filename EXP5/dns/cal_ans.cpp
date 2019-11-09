#include<iostream>
#include<cstdlib>
#include<cmath>
#include<fstream>
#include"omp.h"

#define N 32768

using namespace std;

void init_matrix(double **A, double **B, double **C, double **C2 ,int local_i, int local_j, int block_size);
void init_matrix_s(double **A, double **B, double **C);
void matrix_multi_serial(double **A, double **B, double **C);


int main() {
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
    
    init_matrix_s(A, B, C);
    matrix_multi_serial(A, B, C);
    
    ofstream fout("result.dat", ios::out);
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            fout << C[i][j] << " ";
        }
        fout << "\r\n";
    }
    fout.close();

    return 0;
    
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
