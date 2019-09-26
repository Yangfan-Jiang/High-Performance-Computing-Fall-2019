#include<iostream>

#define M 200
#define N 100
#define S 500
#define block_m 20
#define block_n 20
#define block_s 25

const int block_size_s = int(S/block_s);
const int block_size_m = int(M/block_m);
const int block_size_n = int(N/block_n);

using namespace std;

double A[M][S];
double B[S][N];
double C1[M][N];
double C2[M][N];

void init_matrix() {
    for(int i=0; i<M; i++)
        for(int j=0; j<N; j++) {
            C1[i][j] = 0;
            C2[i][j] = 0;
        }
        
    for(int i=0; i < M; i++)
        for(int j=0; j < S; j++)
            A[i][j] = (i-0.1*j+1)/(i+j+1);
        
    for(int i=0; i < S; i++)
        for(int j=0; j < N; j++)
            B[i][j] = (j-0.2*i+1)*(i+j+1)/(i*i+j*j+1);
}

void matrix_multi() {
    for(int i=0; i < M; i++)
        for(int j=0; j < N; j++)
            for(int s=0; s < S; s++)
                C1[i][j] += A[i][s]*B[s][j];
}

void p_matrix_multi(int num_proc) {
    // [block_m, block_n, block_s]          20 * 2 * 5
    int tmp_m = num_proc / (block_n*block_s);
    int tmp_n = (num_proc / block_s) % block_n;
    int tmp_s = num_proc % block_s;
    for(int i=tmp_m*block_size_m; i < tmp_m*block_size_m+block_size_m; i++)
    for(int j=tmp_n*block_size_n; j < tmp_n*block_size_n+block_size_n; j++)
    for(int s=tmp_s*block_size_s; s < tmp_s*block_size_s+block_size_s; s++)
        C2[i][j] += A[i][s]*B[s][j];
}

void print_ans() {
    for (int i=0; i<M; i++) {
        for(int j=0; j<N; j++) {
            cout.precision(2);
            cout << C1[i][j] << " ";
        }
        cout << endl;
    }   
    cout << "\n\n\n" << endl;
    for (int i=0; i<M; i++) {
        for(int j=0; j<N; j++) {
            cout.precision(2);
            cout << C2[i][j] << " ";
        }
        cout << endl;
    }  
}

bool verify() {
    for(int i=0;i < N; i++)
        for(int j=0; j<M; j++) {
            if(abs(C1[i][j]-C2[i][j]) > 0.0001)
                return false;
        }
    return true;
}

int main()
{
    init_matrix();
    for (int k=0;k < block_m*block_n*block_s; k++)
        p_matrix_multi(k);
    matrix_multi();
    if(verify())
        cout << "verified successfully!" << endl;
    else
        cout << "wrong solution!" << endl;
    return 0;
}