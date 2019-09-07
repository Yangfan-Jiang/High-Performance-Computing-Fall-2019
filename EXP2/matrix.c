#include<stdio.h>
#include<pthread.h>
#include<stdlib.h>

#define M 3
#define N 3
#define K 2
#define NUM_THREADS ((M)*(N))
int A[M][K] = {{1,4}, {2,5}, {3,6}};
int B[K][N] = {{8,7,6}, {5,4,3}};
int C[M][N];

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

int main()
{
    int i,j;
    pthread_t tid[NUM_THREADS];
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    
    for(i = 0; i < M; i++)
        for(j = 0; j < N; j++) {
            v *data = (v*)malloc(sizeof(v));
            data->i = i;
            data->j = j;
            //getOneElem(data);
            pthread_creat(&tid[i*N+j], &atr, getOneElem, (void*) data);
        }
        
    for(i = 0; i < NUM_THREADS; i++)
        pthread_join(tid[i], NULL);
    
    for(i = 0; i < M; i++) {
        for(j = 0; j < N; j++) {
            printf("%d ",C[i][j]);
        }
        printf("\n");
    }
    return 0;
}