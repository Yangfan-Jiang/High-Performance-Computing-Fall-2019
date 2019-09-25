#include<iostream>

#define M 10
#define N 10
#define K 10

using namespace std;

int A[M][K];
int B[K][N];
int C[M][N];

void init_matrix() {
    for(int i=0; i < M; i++)
        for(int j=0; j < K; j++)
            A[i][j] = (i-0.1*j+1)/(i+j+1);
        
    for(int i=0; i < K; i++)
        for(int j=0; j < N; j++)
            B[i][j] = (j-0.2*i+1)(i+j+1)/(i*i+j*j+1);
}

void matrix_multi() {
    for(int i=0; i < N; i++)
        for(int j=0; j < M; j++)
            for(int k=0; k < K; k++)
                C[i][j] += A[i][k]*B[k][j];
}

int main()
{}