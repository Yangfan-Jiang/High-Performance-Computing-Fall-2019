#include<iostream>
#include<cmath>
#include <stdlib.h>
#include <time.h>
#include"omp.h"

using namespace std;

#define N 42000

double x[N];
double y[N];
double **A = new double *[N];

void init_matrix(double **A);
void matrix_multi(double **A);

int main() {
        double Times, Times1;
    clock_t start,finish;
    start=clock();
    for(int i=0; i<N; i++)
        A[i] = new double[N];
    finish=clock();
    Times=(double)(finish-start)/CLOCKS_PER_SEC;
    cout << "allocate memory: " << Times << endl;
    
    start=clock();
    init_matrix(A);
    finish=clock();
        Times=(double)(finish-start)/CLOCKS_PER_SEC;
    cout << "initial time: " << Times << endl;


    start=clock();
    matrix_multi(A);
    finish=clock();
    
    Times=(double)(finish-start)/CLOCKS_PER_SEC;
    //Times1=(double)(finish-start)/CLK_TCK;
    
    cout<<"运行时间(秒)(CLOCKS_PER_SEC): "<<Times<<endl;
    //cout<<"运行时间(秒)(CLK_TCK): "<<Times1<<endl;
    
    return 0;
}

void init_matrix(double **A) {
    #pragma omp parallel for
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) 
            A[i][j] = (i-0.1*j+1)/(i+j+1);
    }
    
    for(int i=0; i<N; i++) {
        x[i] = i/(i*i+1);
        y[i] = 0;
    }
}


void matrix_multi(double **A) {
    for(int i=0; i < N; i++)
        for(int j=0; j < N; j++)
            y[i] += A[i][j] * x[j];
}
