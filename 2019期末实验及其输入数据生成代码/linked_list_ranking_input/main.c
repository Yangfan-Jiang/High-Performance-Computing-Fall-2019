#include <stdio.h>
#include <stdlib.h>

int * gen_linked_list_1(unsigned int N)
{

    int * list=NULL;
    if(NULL!=list)
    {
        free(list);
        list=NULL ;
    }

    if(0==N)
    {
        printf("N is 0, exit\n");
        exit(-1);
    }

    list = (int*) malloc(N*sizeof(int));
    if(NULL==list)
    {
        printf("Can not allocate memory for output array\n");
        exit(-1);
    }

    int i;
    for(i=0; i<N; i++)
        list[i]=i-1;

    return list;
}

int* gen_linked_list_2(unsigned int N)
{
    int * list;

    list = gen_linked_list_1(N);

    int p=N/5;

    int i, temp;

    for(i=0; i<N; i+=2)
    {
        temp = list[i];
        list[i] = list[(i+(i+p))%N];
        list[(i+(i+p))%N] = temp;
    }

    return list;
}


int main()
{
    //int N=1000000;
    int N = 100;
	int* qq=NULL;
    qq=gen_linked_list_1(N);
    int i;
    printf("\nhere is the list\n");
    for(i=0; i<N; i++)
        printf("%3d ", qq[i]);
    printf("\n");
//    printf("%3d ", qq[44]);
    free(qq);
    qq=gen_linked_list_2(N);
    printf("\nhere is the new list\n");
    for(i=0; i<N; i++)
        printf("%3d ", qq[i]);
    printf("\n");
//    printf("%3d ", qq[440]);
    free(qq);

    return 0;

}

