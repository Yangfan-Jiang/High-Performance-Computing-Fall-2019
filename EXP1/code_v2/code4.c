#include<stdio.h>
#define MAX 4096
#define RIDX(i, j, dim) ((i)*(dim) + (j))
#define COPY(d, s) *(d) = *(s)

typedef struct{
    int red; 
    int green; 
    int blue;
}pixel;

void rotate(int dim, pixel *src, pixel *dst)
{
    int i, j;
    for (i=0; i < dim; i+=32)
	for (j=dim-1; j >= 0; j-=1) {
        pixel *dptr = dst+RIDX(dim-1-j,i,dim);
        pixel *sptr = src+RIDX(i,j,dim);
        COPY(dptr   , sptr); sptr += dim; COPY(dptr+1 , sptr); sptr += dim;
        COPY(dptr+2 , sptr); sptr += dim; COPY(dptr+3 , sptr); sptr += dim;
        COPY(dptr+4 , sptr); sptr += dim; COPY(dptr+5 , sptr); sptr += dim;
        COPY(dptr+6 , sptr); sptr += dim; COPY(dptr+7 , sptr); sptr += dim;
        COPY(dptr+8 , sptr); sptr += dim; COPY(dptr+9 , sptr); sptr += dim;
        COPY(dptr+10, sptr); sptr += dim; COPY(dptr+11, sptr); sptr += dim;
        COPY(dptr+12, sptr); sptr += dim; COPY(dptr+13, sptr); sptr += dim;
        COPY(dptr+14, sptr); sptr += dim; COPY(dptr+15, sptr); sptr += dim;
        COPY(dptr+16, sptr); sptr += dim; COPY(dptr+17, sptr); sptr += dim;
        COPY(dptr+18, sptr); sptr += dim; COPY(dptr+19, sptr); sptr += dim;
        COPY(dptr+20, sptr); sptr += dim; COPY(dptr+21, sptr); sptr += dim;
        COPY(dptr+22, sptr); sptr += dim; COPY(dptr+23, sptr); sptr += dim;
        COPY(dptr+24, sptr); sptr += dim; COPY(dptr+25, sptr); sptr += dim;
        COPY(dptr+26, sptr); sptr += dim; COPY(dptr+27, sptr); sptr += dim;
        COPY(dptr+28, sptr); sptr += dim; COPY(dptr+29, sptr); sptr += dim;
        COPY(dptr+30, sptr); sptr += dim; COPY(dptr+31, sptr);
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