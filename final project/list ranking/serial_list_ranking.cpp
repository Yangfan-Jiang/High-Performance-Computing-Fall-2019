#include"gen_list_ranking_data.c" 
#include<iostream>
#include<ctime>
#include<cstdlib>

#define N 100000000

using namespace std;


void list_ranking() {
	int* d = (int*)malloc(N*sizeof(int));
	int* n = NULL;
	n = gen_linked_list_2(N);
	
	// initialize
	for(int i=0; i<N; i++) {
		d[i] = n[i]==-1?0:1;
	}

	int max = -1;
	int cnt = 0;
	while(max<N-1) {
		// parallel
		for(int i=0; i<N; i++) {
			if(n[i] != -1) {
				d[i] = d[i] + d[n[i]];
				n[i] = n[n[i]];
				if(max < d[i]) {
					max = d[i];
				}
			}
		}
		cnt++;
		cout << max << endl;
	}
//cout << "cnt " << cnt << endl;
 //   for(int i=0; i<N; i++)  //????????
 //       printf("%3d ", d[i]);
 //   printf("\n");
	free(n);
    free(d);
}

int main()
{
	clock_t start, end;
	start = clock();
	for(int i=0; i<1; i++)
		list_ranking();
	end = clock();
	double t = (double)(end-start)/CLOCKS_PER_SEC;
	cout << "serial time" << t <<"s" << endl;
	return 0;
}
