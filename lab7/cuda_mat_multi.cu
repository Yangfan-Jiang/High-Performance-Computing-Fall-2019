#include<cuda_runtime_api.h>
#include<device_launch_parameters.h>
#include<stdio.h>
#include<math.h>
#include<time.h>

#define N 16384

__global__ void MatVecMultiply(double a[][N], double b[N], double y[N]) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
    int loc_x = i*128 + j;
    double sum = 0;
	if (loc_x < N)
	{
        for(int i=0; i<N; i++) {
            sum += a[loc_x][i]*b[i];
        }
	}
    y[loc_x] = sum;
}

void matrix_init(double A[][N], double b[N]) {
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            A[i][j] = i-0.1*j+1;
        }
        b[i] = log(sqrt(i*i-i+2));
    }
}

bool verify(double C1[N], double C2[N]) {
    for(int i=0; i<N; i++) {
        if(fabs(C1[i]-C2[i]) > 0.001) {
            printf("%f %f\n",C1[i], C2[i]);
            printf("%d\n",i);
            return false;
        }
    }
    return true;
}

int main()
{   
    clock_t start, finish;
	cudaError_t error = cudaSuccess;

    double (*A)[N] = new double[N][N];
	double *b = new double[N];
    double *C = new double[N];
	double *C2 = new double[N];
	double (*device_a)[N];
    double *device_b, *device_c;

	error = cudaMalloc((void **)&device_a, sizeof(double)* N*N);
	error = cudaMalloc((void **)&device_b, sizeof(double)* N);
	error = cudaMalloc((void **)&device_c, sizeof(double)* N);
   
    matrix_init(A, b);

    //start = clock();
	cudaMemcpy(device_a, A, sizeof(double)* N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, b, sizeof(double)* N, cudaMemcpyHostToDevice);
    
    
    start = clock();
    int TILE_WIDTH = 32;
	dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 numBlocks(4, 4);
	MatVecMultiply << <numBlocks, threadsPerBlock>> >(device_a, device_b, device_c);
	finish = clock();
    cudaMemcpy(C2, device_c, sizeof(double)* N, cudaMemcpyDeviceToHost);
    //finish = clock();
    
    float GPU_Time = (float)(finish - start)/CLOCKS_PER_SEC;
    
    start = clock();
    for(int i=0; i<N; i++) {
        double sum = 0;
        for(int j=0; j<N; j++) {
            sum += A[i][j]*b[j];
        }
        C[i] = sum;
    }
    
    finish = clock();
    float CPU_Time = (float)(finish - start)/CLOCKS_PER_SEC;
    
    if(verify(C, C2)) {
        printf("GPU Time: %fs\n", GPU_Time);
        printf("CPU Time: %fs\n", CPU_Time);
    } else {
        printf("Incorrect Result!!\n");
    }

	return 0;
}
