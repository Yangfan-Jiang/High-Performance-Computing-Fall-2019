#include<iostream>
#include<fstream>
#include<algorithm>
#include<string.h>
#include<string>
#include<vector>
#include<ctime>
#include<cstdlib>
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

clock_t start, tend;

int main()
{
    int Msize1, Msize2, Nsize1, Nsize2;
    int numM, numN;
    
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
    
    
    cnt = 0;
    // split row i into different process
    for(int i=0; i<Msize1; i++) {
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
            ++cnt;
        }
	}
	tend = clock();
    cout << "serial elapsed: " << double(tend-start)/CLOCKS_PER_SEC << "s" << endl;
    
    for(int i=0; i<cnt+2; i++) {
		cout << C[i].row << " " << C[i].col << " " << C[i].value << endl;
    }
    
    return 0;
}
