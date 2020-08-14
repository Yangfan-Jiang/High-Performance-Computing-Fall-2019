#include<iostream>
#include<fstream>
#include<algorithm>
#include<string.h>
#include<string>
#include<vector>
#include<mpi.h>
#include<set>

using namespace std;

typedef struct {
    int row, col;
    double value;
}Triple;

Triple M[1000000];
Triple N[1000000];
Triple C[100000000];

bool cmp(Triple x, Triple y) {
    if(x.row > y.row || ((x.row == y.row) && (x.col > y.col))) 
        return false;
    return true;
}

int main()
{
    int comm_sz, my_rank;
    int start, end;
    
    int Msize1, Msize2, Nsize1, Nsize2;
    int numM, numN;
    
    // MPI_ initialization
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
    // read data
    ifstream fin;
    fin.open("matrix.mat", ios::in);
    string s;
    fin >> s >> s >> s >> s >> s;
    fin >> Msize1 >> Msize2 >> numM;
    
    int cnt = 0;
    int row, col;
    double value;
    while (!fin.eof()) {
        fin >> row >> col >> value;
        M[cnt].row = row;
        M[cnt].col = col;
        M[cnt].value = value;
        ++cnt;
    }
    fin.close();
    
    
    fin.open("matrix12.mat", ios::in);
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
    
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    
    sort(M, M+numM, cmp);
    sort(N, N+numN, cmp);
    
    //cout << "After sort" << endl;
    
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
    
    //cout << "Before while" << endl;
    
    int local_len = Msize1/comm_sz;
    int local_start = my_rank*local_len;
    int local_end = local_start+local_len;
    if(my_rank == comm_sz-1) {
        local_end = Msize1;
    }
    //cout << "myrank:" << my_rank << " start:" << local_start << " end: " << local_end << endl;
    cnt = 0;
    // split row i into different process
    for(int i=local_start; i<local_end; i++) {
        if(indexM[i]==-1)
            continue;
    	for(int j=0; j<Nsize2; j++) {
            if(indexN[j]==-1)
                continue;
            int p=0;
            int q=0;
            double sum = 0.0;
            int flag = 0;
            while(p<lenM[i] && q<lenN[j]) {
                if(M[p+indexM[i]].col == N[q+indexN[j]].col) {
                	flag = 1;
                    sum += M[p+indexM[i]].value*N[q+indexN[j]].value;
                    ++p;++q;
                }
                else if(M[p+indexM[i]].col < N[q+indexN[j]].col) {
                    ++p;
                }
                else {
                    ++q;
                }
            }
            if(!flag)
            	continue;
            C[cnt].row = i;
            C[cnt].col = j;
            C[cnt].value = sum;
            //cout << cnt << endl;
            ++cnt;
        }
        //cout << i << " " << cnt << endl;
	}
	MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    if(my_rank == 0) {
        cout << "num of proc: " << comm_sz << endl;
        cout << "MPI elapsed: " << end-start << "s" << endl;
    }
    MPI_Finalize();
    
    return 0;
}
