#include<stdio.h>
#define MAX 512
#define RIDX(i, j, dim) ((i)*(MAX) + (j)*(MAX) + (dim))

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

pixel src[MAX*MAX*MAX];
pixel dst[MAX*MAX*MAX];

int main() 
{
    naive_rotate(MAX, src, dst);
    return 0;
}