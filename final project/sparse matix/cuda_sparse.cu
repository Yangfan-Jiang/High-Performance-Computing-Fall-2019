#include<iostream>
#include<fstream>
#include<algorithm>
#include<string.h>
#include<string>
#include<ctime>
#include<cstdlib>

#include<cuda_runtime_api.h>
#include<device_launch_parameters.h>

#define block_size 4
#define thread_size 32

using namespace std;

//const int total_threads = block_size*block_size*thread_size*thread_size;

typedef struct {
    int row, col;
    float value;
}Triple;

Triple M[1000000];
Triple N[1000000];
//Triple C[100000000];
float C[9000][9000];

int rowM[1000000];
int colM[1000000];
float valueM[1000000];
int rowN[1000000];
int colN[1000000];
float valueN[1000000];


bool cmp(Triple x, Triple y) {
    if(x.row > y.row || ((x.row == y.row) && (x.col > y.col))) 
        return false;
    return true;
}

__global__ void cudaMat(float valueM[], float valueN[], 
                        int rowM[], int rowN[],
                        int colM[], int colN[],
                        int indexM[], int indexN[],
                        int lenM[], int lenN[],
                        float C[][9000], int size2[]
                        ) 
{
    int size = size2[0];
    size = 8191;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
    int my_rank = i*thread_size*block_size + j;
    
    i = my_rank/2;
    if(i >= size)
        return;
    
    if(indexM[i]==-1)
        return;
    
    int jstart = 0;
    int jend = size/2;
    if(my_rank%2) {
        jstart = size/2;
        jend = size;
    }
    for(int j=jstart; j<jend; j++) {
        if(indexN[j]==-1)
            continue;
        int p=0;
        int q=0;
        float sum = 0.0;
        int flag = 0;
        
        while(p<lenM[i] && q<lenN[j]) {
            if(colM[p+indexM[i]] == colN[q+indexN[j]]) {
                flag = 1;
                sum += valueM[p+indexM[i]] *valueN[q+indexN[j]];
                ++p;++q;
            }
            else if(colM[p+indexM[i]] < colN[q+indexN[j]]) {
                ++p;
            }
            else {
                ++q;
            }
        }
        if(!flag)
            continue;
        C[i][j] = sum;
    }
    
}


clock_t start, tend;

int main()
{
    int Msize1, Msize2, Nsize1, Nsize2;
    int numM, numN;
    
    // read data
    ifstream fin;
    fin.open("matrix.mat", ios::in);
    //fin.open("test.dat", ios::in);
    string s;
    fin >> s >> s >> s >> s >> s;
    fin >> Msize1 >> Msize2 >> numM;
    
    int cnt = 0;
    int row, col;
    float value;
    while (!fin.eof()) {
        fin >> row >> col >> value;
        M[cnt].row = row;
        M[cnt].col = col;
        M[cnt].value = value;
        ++cnt;
    }
    fin.close();
    
    
    fin.open("matrix12.mat", ios::in);
    //fin.open("test2.dat", ios::in);
    fin >> s >> s >> s >> s >> s;
    fin >> Nsize1 >> Nsize2 >> numN;
    cnt = 0;
    while (!fin.eof()) {
        fin >> row >> col >> value;
        N[cnt].row = col;
        N[cnt].col = row;
        N[cnt].value = value;
        ++cnt;
    }
    fin.close();
    
    start = clock();
    
    sort(M, M+numM, cmp);
    sort(N, N+numN, cmp);
    
    int indexM[Msize1+1];
    int indexN[Nsize2+1];
    int lenM[Msize1+1];
    int lenN[Nsize2+1];
    memset(indexM, -1, Msize1*sizeof(int));
    memset(indexN, -1, Nsize2*sizeof(int));
    cnt = 0;
    
    int k = -1;
    for(int i=0; i<numM; i++) {
        if(indexM[M[i].row] == -1) {
            indexM[M[i].row] = i;
            k = M[i].row;
            lenM[k] = 1;
            while(M[i+1].row == k) {
            	i++;
            	lenM[k]++;
			}
        }
    }
    for(int i=0; i<numM; i++) {
        rowM[i] = M[i].row;
        colM[i] = M[i].col;
        valueM[i] = M[i].value;
    }
    
    for(int i=0; i<numN; i++) {
        if(indexN[N[i].row] == -1) {
            indexN[N[i].row] = i;
            k = N[i].row;
            lenN[k] = 1;
            while(N[i+1].row == k) {
            	i++;
            	lenN[k]++;
			}
        }
    }
    for(int i=0; i<numN; i++) {
        rowN[i] = N[i].row;
        colN[i] = N[i].col;
        valueN[i] = N[i].value;
    }

    cnt = 0;
    // split row i into different process
    int *device_rowM, *device_colM, *device_rowN, *device_colN, *device_indexM, *device_indexN, *device_lenM, *device_lenN;
    float *device_valueM, *device_valueN;
    float (*device_C)[9000];
    int *device_size;
    
    cudaMalloc((void **)&device_C, sizeof(float)* 9000*9000);
    cudaMalloc((void **)&device_rowM, sizeof(int)* 1000000);
    cudaMalloc((void **)&device_colM, sizeof(int)* 1000000);
    cudaMalloc((void **)&device_rowN, sizeof(int)* 1000000);
    cudaMalloc((void **)&device_colN, sizeof(int)* 1000000);
    cudaMalloc((void **)&device_indexM, sizeof(int)* 1000000);
    cudaMalloc((void **)&device_indexN, sizeof(int)* 1000000);
    cudaMalloc((void **)&device_lenM, sizeof(int)* 1000000);
    cudaMalloc((void **)&device_lenN, sizeof(int)* 1000000);
    
    cudaMalloc((void **)&device_valueM, sizeof(float)* 1000000);
    cudaMalloc((void **)&device_valueN, sizeof(float)* 1000000);
    cudaMalloc((void **)&device_size, sizeof(int));
    
    //cudaMemcpy(device_C, C, sizeof(float)* N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_rowM, rowM, sizeof(int)* 1000000, cudaMemcpyHostToDevice);
    cudaMemcpy(device_colM, colM, sizeof(int)* 1000000, cudaMemcpyHostToDevice);
    cudaMemcpy(device_rowN, rowN, sizeof(int)* 1000000, cudaMemcpyHostToDevice);
    cudaMemcpy(device_colN, colN, sizeof(int)* 1000000, cudaMemcpyHostToDevice);
    
    cudaMemcpy(device_indexM, indexM, sizeof(int)* Msize1+1, cudaMemcpyHostToDevice);
    cudaMemcpy(device_indexN, indexN, sizeof(int)* Msize1+1, cudaMemcpyHostToDevice);
    cudaMemcpy(device_lenM, lenM, sizeof(int)* Msize1+1, cudaMemcpyHostToDevice);
    cudaMemcpy(device_lenN, lenN, sizeof(int)* Msize1+1, cudaMemcpyHostToDevice);
    cudaMemcpy(device_valueM, valueM, sizeof(float)* 1000000, cudaMemcpyHostToDevice);
    cudaMemcpy(device_valueN, valueN, sizeof(float)* 1000000, cudaMemcpyHostToDevice);
    cudaMemcpy(device_size, &Msize1, sizeof(int), cudaMemcpyHostToDevice);
    
	//int max = -1;
    dim3 threadsPerBlock(thread_size, thread_size);
    dim3 numBlocks(4, 4);

    cudaMat <<< numBlocks, threadsPerBlock >>>(device_valueM, device_valueN,
                                               device_rowM, device_rowN,
                                               device_colM, device_colN,
                                               device_indexM, device_indexN,
                                               device_lenM, device_lenN,
                                               device_C, device_size);
    cudaMemcpy(C, device_C, sizeof(float)* 9000*9000, cudaMemcpyDeviceToHost);
	tend = clock();

    cout << "serial elapsed: " << float(tend-start)/CLOCKS_PER_SEC << "s" << endl;
    
    return 0;
}
