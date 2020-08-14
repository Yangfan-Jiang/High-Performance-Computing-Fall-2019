#include<cuda_runtime_api.h>
#include<device_launch_parameters.h>
#include<stdio.h>
#include<math.h>
#include<time.h>
#include<stdlib.h>

#define N 4096


__global__ void MatMulti(const float *a, const float *b, float *result)
{
    int threadId = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x 
                    + blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId < N * N)
    {
        int row = threadId / N;
        int column = threadId % N;
 
        result[threadId] = 0;
        for (int i = 0; i < N; i++)
        {
            result[threadId] += a[row * N + i] * b[i * N + column];
        }
    }
}


void matrix_init(float *A, float *B) {
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            A[i*N + j] = (i-0.1*j+1)/(i+j+1);
            B[i*N + j] = (j-0.2*i+1)*(i+j+1)/(i*i+j*j+1);
        }
    }
}


bool verify(float C1[N], float C2[N]) {
    for(int i=0; i<N; i++) {
        if(fabs(C1[i]-C2[i])/C1[i] > 0.001) {
            printf("%f %f\n",C1[i], C2[i]);
            printf("%d\n",i);
            return false;
        }
    }
    return true;
}

int main()
{   
    float GPU_Time;
    cudaEvent_t start, stop;
    
	cudaError_t error = cudaSuccess;
    float *A = new float[N*N];
	float *B = new float[N*N];
    float *C = new float[N*N];
    
    float *device_a, *device_b, *device_c;

	error = cudaMalloc((void **)&device_a, sizeof(float)* N*N);
	error = cudaMalloc((void **)&device_b, sizeof(float)* N*N);
	error = cudaMalloc((void **)&device_c, sizeof(float)* N*N);
   
    matrix_init(A, B);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
	cudaMemcpy(device_a, A, sizeof(float)* N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, B, sizeof(float)* N*N, cudaMemcpyHostToDevice);
    
    int TILE_WIDTH = 32;
	dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 numBlocks(128, 128);
    
    cudaEventRecord(start, 0);    
	MatMulti << <numBlocks, threadsPerBlock>> >(device_a, device_b, device_c);
    cudaEventRecord(stop, 0);
    
    cudaMemcpy(C, device_c, sizeof(float)* N*N, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(start);    //等待事件完成。
    cudaEventSynchronize(stop);    //等待事件完成。记录之前的任务
    cudaEventElapsedTime(&GPU_Time, start, stop);

    cudaEventDestroy(start);    //消除Event
    cudaEventDestroy(stop);
    // read cpu result
    float *C2 = new float[N*N];
    FILE* fp;
    fp = fopen("cpu_result", "r");
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            fscanf(fp, "%f ", &C2[i*N + j]);
        }
    }
    fclose(fp);
    
    if(verify(C, C2)) {
        printf("GPU Time: %fms\n", GPU_Time);
    }
    
	return 0;
}
