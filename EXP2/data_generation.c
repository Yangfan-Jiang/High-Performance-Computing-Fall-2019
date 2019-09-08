#include<stdio.h>
#include<stdlib.h>

int M, N;

int main(int argc, char* argv[])
{
    int i, j, num;
    FILE* fp;
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    
    fp = fopen("data", "w");
    fprintf(fp, "%d %d\n", M, N);
    for(i=0; i<M; i++) {
        for(j=0; j<N; j++) {
            num = rand()%1000;
            fprintf(fp, "%d\n", i, j, num);
        }
    }
    fclose(fp);
    return 0;
}