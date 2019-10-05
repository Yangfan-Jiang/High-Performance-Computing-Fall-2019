#include<iostream>
#include<pthread.h>
#include<chrono>
#include<ctime>

#define M 2800
#define N 2800
#define S 2100
#define block_m 4
#define block_n 4
#define block_s 3
#define THREAD_NUM (block_m)*(block_n)*(block_s)

const int block_size_s = int(S/block_s);
const int block_size_m = int(M/block_m);
const int block_size_n = int(N/block_n);

using namespace std;
using namespace chrono;

double A[M][S];
double B[S][N];
double C1[M][N];
double C2[M][N];
double private_C[block_m*block_n*block_s][block_size_m][block_size_n];

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

void* p_matrix_multi(void *k) {
    // [block_m, block_n, block_s]          20 * 2 * 5
    int num_proc = *(int *)k;

    int tmp_m = num_proc / (block_n*block_s);
    int tmp_n = (num_proc / block_s) % block_n;
    int tmp_s = num_proc % block_s;

    for(int i=tmp_m*block_size_m; i < tmp_m*block_size_m+block_size_m; i++)
    for(int j=tmp_n*block_size_n; j < tmp_n*block_size_n+block_size_n; j++)
    for(int s=tmp_s*block_size_s; s < tmp_s*block_size_s+block_size_s; s++) {
        private_C[num_proc][i-tmp_m*block_size_m][j-tmp_n*block_size_n] += A[i][s]*B[s][j];
    }
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

void reduction() {
    for(int num_proc=0; num_proc<THREAD_NUM; num_proc++){
        int tmp_m = num_proc / (block_n*block_s);
        int tmp_n = (num_proc / block_s) % block_n;
        int tmp_s = num_proc % block_s;
        for(int i=tmp_m*block_size_m; i < tmp_m*block_size_m+block_size_m; i++)
        for(int j=tmp_n*block_size_n; j < tmp_n*block_size_n+block_size_n; j++)
            C2[i][j] += private_C[num_proc][i-tmp_m*block_size_m][j-tmp_n*block_size_n];
    }
}

bool verify() {
    for(int i=0;i < N; i++)
        for(int j=0; j < M; j++) {
            if(abs(C1[i][j]-C2[i][j]) > 0.0001)
                return false;
        }
    return true;
}

int main()
{
    init_matrix();
    
    pthread_t tid[THREAD_NUM];
    pthread_attr_t attr;
    pthread_attr_init(&attr);

    int thread_id[THREAD_NUM];
    for(int i=0; i<THREAD_NUM; i++)
        thread_id[i]=i;
    
    steady_clock::time_point start = steady_clock::now();
    for (int k=0; k < THREAD_NUM; k++) {
        pthread_create(&tid[k], NULL, p_matrix_multi, &thread_id[k]);
    }
        //p_matrix_multi(&k);
    for (int k=0; k < THREAD_NUM; k++)
        pthread_join(tid[k], NULL);
    
    reduction();
    steady_clock::time_point end = steady_clock::now();
    steady_clock::duration d = end-start;
    cout << "parallel: " << duration_cast<microseconds>(d).count()/1000000.0 << "s" << endl;
    
    
    start = steady_clock::now();
    
    matrix_multi();
    
    end = steady_clock::now();
    d = end-start;
    cout << "serial: " << duration_cast<microseconds>(d).count()/1000000.0 << "s" << endl;
    
    if(verify())
        cout << "verified successfully!" << endl;
    else
        cout << "wrong solution!" << endl;
    return 0;
}