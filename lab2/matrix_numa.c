#include<stdio.h>
#include<pthread.h>
#include<stdlib.h>

#define K 2
#define NUM_THREADS 32

//#define _GNU_SOURCE

int M, N;
int num_cpu = 1;

int **A;
int *x;
int **y;
int **B;
int **C;

typedef struct{
   int i;
   int j;
}v;

void* getOneElem(void *data) {
    int i,j,k;
    v *d = (v*)data;
    i = d->i;
    j = d->j;
    for(k = 0; k < K; k++)
        C[i][j] += A[i][k]*B[k][j];
}

void init() {    
    int i, j, k;
    FILE* fp = fopen("data", "r");
    fscanf(fp, "%d%d", &M, &N);
    A = (int**)malloc(sizeof(int*)*M);
    x = (int*)malloc(sizeof(int)*N);
    y = (int**)malloc(sizeof(int*)*M);

    // 填充向量y,每个进程读16个int
    for(i=0; i<M; i++) {
        A[i] = (int*)malloc(sizeof(int)*N);
        y[i] = (int*)malloc(sizeof(int)*8);
    }
    for(i=0; i<M; i++) {
        for(j=0; j<N; j++) {
            fscanf(fp, "%d", &A[i][j]);
        }
    }
    for(i=0; i<N; i++) {
        x[i] = rand()%1000;
    }
    fclose(fp);
}


inline int set_cpu(int i) {
    cpu_set_t mask;  
    CPU_ZERO(&mask);  
  
    CPU_SET(i,&mask);  
  
    //printf("thread %u, i = %d\n", pthread_self(), i);  
    if(-1 == pthread_setaffinity_np(pthread_self() ,sizeof(mask),&mask))  
    {  
        fprintf(stderr, "pthread_setaffinity_np erro\n");  
        return -1;  
    }  
}


void* multiply(void *data) {
    int start, end, i, j;
    
    if (set_cpu(++num_cpu)) {
        return NULL;
    }
    
    v *d = (v*)data;
    start = d->i;
    end = d->j;
    for(i=start; i<=end; i++) {
        for(j=0; j<N; j++) {
            y[i][0] += x[j]*A[i][j];
        }
    }
}


int main()
{
    int i,j;
    pthread_t tid[NUM_THREADS];
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    
    init();
    
    int k;
    double start = clock();
    for(k=0; k<100; k++) {
        num_cpu = 1;
        for(i=0; i < NUM_THREADS; i++) {
            v *data = (v*)malloc(sizeof(v));
            data->i = i*(M / NUM_THREADS);
            data->j = (i+1)*(M / NUM_THREADS)-1;
            pthread_create(&tid[i], &attr, multiply, (void*) data);
        }
        
        for(i = 0; i < NUM_THREADS; i++)
            pthread_join(tid[i], NULL);
    }
    double end = clock();
    double time = (double)(end - start)/CLOCKS_PER_SEC;
    printf("%lf\n",time);
    FILE *fp = fopen("result_numa", "w");
    for (i=0; i<M; i++) {
        fprintf(fp, "%d 0 %d\n", i, y[i][0]);
    }
    fclose(fp);
    
    return 0;
}
