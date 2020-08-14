#include<cuda_runtime_api.h>
#include<device_launch_parameters.h>
#include<stdio.h>
#include<math.h>
#include<time.h>

#define N 8192

float **A;
float **B;
float **C;
float **C2;
float **device_a;
float **device_b;
float **device_c;

void matrix_init() {
    A = (float **)malloc(N*sizeof(float *));
    B = (float **)malloc(N*sizeof(float *));
    C = (float **)malloc(N*sizeof(float *));
    C2 = (float **)malloc(N*sizeof(float *));
    device_a = (float **)malloc(N*sizeof(float *));
    device_b = (float **)malloc(N*sizeof(float *));
    device_c = (float **)malloc(N*sizeof(float *));
    
    for(int i=0; i<N; i++) {
        A[i] = (float*)malloc(N*sizeof(float));
        B[i] = (float*)malloc(N*sizeof(float));
        C[i] = (float*)malloc(N*sizeof(float));
        C2[i] = (float*)malloc(N*sizeof(float));
        device_a = (float **)malloc(N*sizeof(float *));
        device_b = (float **)malloc(N*sizeof(float *));
        device_c = (float **)malloc(N*sizeof(float *));
    }
    
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            A[i][j] = i-0.1*j+1;
            B[i][j] = 0.2*j-0.1*i;
        }
    }
} 

__global__ void MatAdd(float **A, float **B, float **C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N || i<0 || j<0) {
        printf("In GPU: %d %d\n",i, j);
    }
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}

bool verify(float **C1, float **C2) {
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            if(fabs(C1[i][j]-C2[i][j]) > 0.001) {
                printf("%d %d\n",i, j);
                return false;
            }
        }
    }
    return true;
}

int main()
{
    matrix_init();
    clock_t start, finish;
    
	//float (*device_a)[N],(*device_b)[N],(*device_c)[N];
    cudaError_t error = cudaSuccess;
 
	error = cudaMalloc((void **)&device_a, sizeof(float)* N*N);
	error = cudaMalloc((void **)&device_b, sizeof(float)* N*N);
	error = cudaMalloc((void **)&device_c, sizeof(float)* N*N);
    
    cudaMemcpy(device_a, A, sizeof(float)* N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, B, sizeof(float)* N*N, cudaMemcpyHostToDevice);
    
    start = clock();
    int TILE_WIDTH = 16;
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(device_a, device_b, device_c);
    printf("test")
    //cudaDeviceSynchronize();
    finish = clock();
    
    cudaMemcpy(C2, device_c, sizeof(float)* N*N, cudaMemcpyDeviceToHost);
    
    double GPU_Time = (double)(finish - start)/CLOCKS_PER_SEC;
   
    start = clock();
    
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            C2[i][j] = A[i][j] + B[i][j];
        }
    }
    
    finish = clock();
    double CPU_Time = (double)(finish - start)/CLOCKS_PER_SEC;
    
    if(verify(C, C2)) {
        printf("GPU Time: %fs\n", GPU_Time);
        printf("CPU Time: %fs\n", CPU_Time);
    } else {
        printf("Incorrect Result!!");
    }
    
    return 0;
}
