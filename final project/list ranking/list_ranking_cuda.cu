#include"gen_list_ranking_data.c" 
#include<iostream>
#include<cuda_runtime_api.h>
#include<cmath>
#include<device_launch_parameters.h>

#define N 100000000
#define block_size 4
#define thread_size 32

using namespace std;

const int total_threads = block_size*block_size*thread_size*thread_size;

__global__ void cuda_list_ranking(int n[], int d[]) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
    int my_rank = i*thread_size*block_size + j;
    int local_len = N/total_threads;
    int local_start = my_rank*local_len;
    int local_end = local_start + local_len;
    if(my_rank == total_threads-1) {
        local_end = N;
    }
    for(int i = local_start; i<local_end; i++) {
        if(n[i] != -1) {
            d[i] = d[i] + d[n[i]];
            n[i] = n[n[i]];
        }
    }
}

int get_max(int* d){
    int max = -1;
    for(int i=0; i<N; i++) {
        max = (max < d[i])?d[i]:max;
    }
    return max;
}

int main()
{
    clock_t start, finish;
    int* d = (int*)malloc(N*sizeof(int));
	int* n = NULL;
	n = gen_linked_list_2(N);
	
    start = clock();
    int* device_d;
    int* device_n;
    
    cudaMalloc((void **)&device_d, sizeof(int)* N);
    cudaMalloc((void **)&device_n, sizeof(int)* N);
    
	// initialize
	for(int i=0; i<N; i++) {
		d[i] = n[i]==-1?0:1;
	}
    
    cudaMemcpy(device_d, d, sizeof(float)* N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_n, n, sizeof(float)* N, cudaMemcpyHostToDevice);
    
	//int max = -1;
    dim3 threadsPerBlock(thread_size, thread_size);
    dim3 numBlocks(4, 4);
    //int cnt = 0;
	while(get_max(d) < N-1) {
        // parallel
        cuda_list_ranking <<<numBlocks, threadsPerBlock>>>(device_n, device_d);
        cudaMemcpy(d, device_d, sizeof(float)* N, cudaMemcpyDeviceToHost);
	}
    
    cudaMemcpy(d, device_d, sizeof(float)* N, cudaMemcpyDeviceToHost);
    cudaMemcpy(n, device_n, sizeof(float)* N, cudaMemcpyDeviceToHost);
	finish = clock();
    cout << "GPU time: " << double(finish-start)/CLOCKS_PER_SEC << "s" << endl;
    free(n);
    free(d);
	return 0;
}

