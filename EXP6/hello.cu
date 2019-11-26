#include<stdio.h>


__global__ void device_greetings() 
{
    printf("Hello, world from the GPU!\n");
}

int main()
{
    printf("Hello, world form the host!\n");

    dim3 threadBlocks(8, 16);
    dim3 gridBlocks(2, 4);
    device_greetings<<<gridBlocks, threadBlocks>>>();
    cudaDeviceSynchronize();

    return 0;
}