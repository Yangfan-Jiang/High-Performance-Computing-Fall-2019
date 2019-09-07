#include<stdio.h>
#define MAX 4096
#define RIDX(i, j, dim) ((i)*(dim) + (j))

typedef struct{
    int red; 
    int green; 
    int blue;
}pixel;

void naive_rotate(int dim, pixel *src, pixel *dst)
{
    int i, j;
    for (i = 0; i < dim; i++)
    for (j = 0; j < dim; j++)
        dst[RIDX(dim-1-j, i, dim)] = src[RIDX(i, j, dim)];
}

pixel src[MAX*MAX];
pixel dst[MAX*MAX];

int main() 
{
    int i;
    for (i = 0; i < 20; i++)
        naive_rotate(MAX, src, dst);
    return 0;
}