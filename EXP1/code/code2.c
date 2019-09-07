#include<stdio.h>
#define MAX 4096
#define RIDX(i, j, dim) ((i)*(dim) + (j))

typedef struct{
    int red; 
    int green; 
    int blue;
}pixel;

void rotate(int dim, pixel *src, pixel *dst)
{
    int i, j, ii, jj;
    for (ii = 0; ii < dim; ii+=4)
    for (jj = 0; jj < dim; jj+=4)
        for (i=ii; i < ii+4; i++)
        for (j=jj; j < jj+4; j++)
            dst[RIDX(dim-1-j, i, dim)] = src[RIDX(i, j, dim)];
}

pixel src[MAX*MAX];
pixel dst[MAX*MAX];

int main() 
{
    int i;
    for (i = 0; i < 20; i++)
        rotate(MAX, src, dst);
    return 0;
}