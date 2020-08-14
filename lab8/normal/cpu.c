#include<stdio.h>
#include<math.h>
#include<time.h>
#include<stdlib.h>

#define N 4096

void matrix_init(float A[][N], float B[][N]) {
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            A[i][j] = (i-0.1*j+1)/(i+j+1);
            B[i][j] = (j-0.2*i+1)*(i+j+1)/(i*i+j*j+1);
        }
    }
}

int main()
{   
    clock_t start, finish;

    float (*A)[N] = new float[N][N];
	float (*B)[N] = new float[N][N];
    float (*C)[N] = new float[N][N];

   
    matrix_init(A, B);

    start = clock();
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            float sum = 0;
            for(int k=0; k<N; k++) {
                sum += A[i][k]*B[k][j];
            }
            C[i][j] = sum;
        }
    }
    
    finish = clock();
    float CPU_Time = (float)(finish - start)/CLOCKS_PER_SEC;
    printf("CPU time: %fs\n", CPU_Time);
    
    FILE* fp;
    fp = fopen("cpu_result", "w");
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            fprintf(fp, "%f ", C[i][j]);
        }
        fprintf(fp,"\n");
    }
    fclose(fp);

	return 0;
}
