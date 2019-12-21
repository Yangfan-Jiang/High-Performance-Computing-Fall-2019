#include<cuda_runtime_api.h>
#include<device_launch_parameters.h>
#include<stdio.h>
#include<math.h>
#include<time.h>

#define N 16384
#define TILE_WIDTH 32

//__constant__ float cuda_A[N][N];
//__constant__ float cuda_c[N];

__global__ void MatVecMultiply(float a[][N], float b[N], float y[N]) {
    //__shared__ float tmp_mat[N];
    __shared__ float tmp_vec[N/2];
    
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
    int loc_x = i*128 + j;
    
    for(int i=0; i<N/2; i++) {
        tmp_vec[i] = b[i];
    }
    
    // init
    if(loc_x < N) {
        float sum = 0;
        for(int i=0; i<N/2; i++) {
            sum += a[loc_x][i]*tmp_vec[i];
        }
        
        for(int i=N/2; i<N; i++) {
            sum += a[loc_x][i]*b[i];
        }
        y[loc_x] = sum;
	}
}

void matrix_init(float A[][N], float b[N]) {
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            A[i][j] = i-0.1*j+1;
        }
        b[i] = log(sqrt(i*i*1.0-i+2.0));
    }
}

bool verify(float C1[N], float C2[N]) {
    for(int i=0; i<N; i++) {
        if(fabs(C1[i]-C2[i])/C1[i] > 0.0001) {
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

    float (*A)[N] = new float[N][N];
	float *b = new float[N];
    float *C = new float[N];
	float *C2 = new float[N];
	float (*device_a)[N];
    float *device_b, *device_c;

	error = cudaMalloc((void **)&device_a, sizeof(float)* N*N);
	error = cudaMalloc((void **)&device_b, sizeof(float)* N);
	error = cudaMalloc((void **)&device_c, sizeof(float)* N);
   
    matrix_init(A, b);
    
    start = clock();
	cudaMemcpy(device_a, A, sizeof(float)* N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, b, sizeof(float)* N, cudaMemcpyHostToDevice);
    
	dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 numBlocks(4, 4);
	MatVecMultiply << <numBlocks, threadsPerBlock>> >(device_a, device_b, device_c);
    cudaMemcpy(C2, device_c, sizeof(float)* N, cudaMemcpyDeviceToHost);
    finish = clock();
    
    float GPU_Time = (float)(finish - start)/CLOCKS_PER_SEC;

    start = clock();
    for(int i=0; i<N; i++) {
        float sum = 0;
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
