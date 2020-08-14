#include<iostream>
#include<set>
#include"gen_list_ranking_data.c"

using namespace std;

int main()
{
    int N = 21;
    int *list = NULL;
    list = gen_linked_list_2(N);
    
    set<int> s;
    // find the start node
    int tmp = -1;
    int lasti = -1;
    for(int i=0; i<N; ++i) {
        if(s.find(i) != s.end())
            continue;
        tmp = i;
        lasti = i;
        while(s.find(tmp) == s.end()) {
            s.insert(tmp);
            if(list[tmp] == -1) {
                s.insert(-1);
                break;
            }
            tmp = list[tmp];
        }
    }
    //printf("%d\n",tmp);
    // count rank of each nodes
    int* rank = (int*) malloc(N*sizeof(int));
    int r = 1;
    tmp = lasti;
    while(tmp != -1) {
        rank[tmp] = r++;
        tmp = list[tmp];
    }
    
    // test
    /*
    for(int i=0; i<N; i++) {
        printf("%d ", list[i]);
    }
    printf("\n");
    
    printf("==========\n");
    for(int i=0; i<N; i++) {
        printf("%d ", rank[i]);
    }
    printf("\n");
    
    tmp = lasti;
    for(int i=0; i<N; i++) {
		printf("%d ", list[tmp]);
		tmp = list[tmp];
	}
    */
    
    free(list);
    free(rank);
    
    return 0;
}
