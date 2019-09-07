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
    for (ii = 0; ii < dim; ii+=32)
    for (jj = 0; jj < dim; jj+=32)
        for (i=ii; i < ii+32; i+=4)
        for (j=jj; j < jj+32; j+=4) {
            dst[RIDX(dim-1-j, i, dim)]   = src[RIDX(i, j, dim)];
            dst[RIDX(dim-1-j, i+1, dim)] = src[RIDX(i+1, j, dim)];
            dst[RIDX(dim-1-j, i+2, dim)] = src[RIDX(i+2, j, dim)];
            dst[RIDX(dim-1-j, i+3, dim)] = src[RIDX(i+3, j, dim)];
            
            dst[RIDX(dim-1-j-1, i, dim)]   = src[RIDX(i, j+1, dim)];
            dst[RIDX(dim-1-j-1, i+1, dim)] = src[RIDX(i+1, j+1, dim)];
            dst[RIDX(dim-1-j-1, i+2, dim)] = src[RIDX(i+2, j+1, dim)];
            dst[RIDX(dim-1-j-1, i+3, dim)] = src[RIDX(i+3, j+1, dim)];
            
            dst[RIDX(dim-1-j-2, i, dim)]   = src[RIDX(i, j+2, dim)];
            dst[RIDX(dim-1-j-2, i+1, dim)] = src[RIDX(i+1, j+2, dim)];
            dst[RIDX(dim-1-j-2, i+2, dim)] = src[RIDX(i+2, j+2, dim)];
            dst[RIDX(dim-1-j-2, i+3, dim)] = src[RIDX(i+3, j+2, dim)];
            
            dst[RIDX(dim-1-j-3, i, dim)]   = src[RIDX(i, j+3, dim)];
            dst[RIDX(dim-1-j-3, i+1, dim)] = src[RIDX(i+1, j+3, dim)];
            dst[RIDX(dim-1-j-3, i+2, dim)] = src[RIDX(i+2, j+3, dim)];
            dst[RIDX(dim-1-j-3, i+3, dim)] = src[RIDX(i+3, j+3, dim)];
        }
}


int main() 
{
    pixel *src = (pixel*)malloc(MAX*MAX*sizeof(pixel));
    pixel *dst = (pixel*)malloc(MAX*MAX*sizeof(pixel));
    int i;
    for (i = 0; i < 20; i++)
        rotate(MAX, src, dst);
    return 0;
}