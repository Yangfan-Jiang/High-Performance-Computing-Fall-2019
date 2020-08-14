#include<cuda_runtime_api.h>
#include<device_launch_parameters.h>
#include<stdio.h>
#include<math.h>
#include<time.h>

#define N 8192

__global__ void add(float a[][N], float b[][N], float c[][N]) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i < N && j < N)
	{
		c[i][j] = a[i][j] + b[i][j];
	}
}

void matrix_init(float A[][N], float B[][N]) {
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            A[i][j] = i-0.1*j+1;
            B[i][j] = 0.2*j-0.1*i;
        }
    }
}

bool verify(float C1[][N], float C2[][N]) {
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            if(fabs(C1[i][j]-C2[i][j]) > 0.001) {
                printf("%f %f\n",C1[i][j], C2[i][j]);
                printf("%d %d\n",i, j);
                return false;
            }
        }
    }
    return true;
}

int main()
{   
    clock_t start, finish;
	cudaError_t error = cudaSuccess;
 
    float (*A)[N] = new float[N][N];
	float (*B)[N] = new float[N][N];
    float (*C)[N] = new float[N][N];
	float (*C2)[N] = new float[N][N];
	float (*device_a)[N],(*device_b)[N],(*device_c)[N];
 
	error = cudaMalloc((void **)&device_a, sizeof(float)* N*N);
	error = cudaMalloc((void **)&device_b, sizeof(float)* N*N);
	error = cudaMalloc((void **)&device_c, sizeof(float)* N*N);
    
    matrix_init(A, B);
 
	cudaMemcpy(device_a, A, sizeof(float)* N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, B, sizeof(float)* N*N, cudaMemcpyHostToDevice);
    
    start = clock();
    int TILE_WIDTH = 32;
	dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
	add << <numBlocks, threadsPerBlock>> >(device_a, device_b, device_c);
    finish = clock();

	cudaMemcpy(C2, device_c, sizeof(float)* N*N, cudaMemcpyDeviceToHost);
    double GPU_Time = (double)(finish - start)/CLOCKS_PER_SEC;
    
    start = clock();
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            C[i][j] = A[i][j] + B[i][j];
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
