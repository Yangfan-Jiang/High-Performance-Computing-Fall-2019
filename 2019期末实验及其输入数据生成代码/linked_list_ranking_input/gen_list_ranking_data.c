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

//i和j的后继元素交换位置
void swap(int * list, int i, int j){
	if(i < 0 || j < 0 || i==j)return;
	
	int p = list[i];  //保存i后继元素下标p
	int q = list[j];  //保存j后继元素下标q

	if(p == -1 || q == -1)return; //如果有一个没有后继元素
	
	int pnext = list[p];  //保存p的后继元素下标
	int qnext = list[q];  //保存q的后继元素下标

	//i,j的后继元素交换位置
	if(p == j){  //j是i的后继
		list[i] = q;
		list[j] = list[q];
		list[q] = j;
	}
	else if(i == q){  //i是j的后继
		list[j] = p;
		list[i] = list[p];
		list[p] = i;
	}
	else{
		list[i] = q;   //i的后继改为q
		list[j] = p;   //j的后继改为p
		list[p] = qnext; //p的后继元素改为原来q的后继	
		list[q] = pnext; //q的后继元素改为原来p的后继
	}
	
}

int* gen_linked_list_2(unsigned int N)
{
    int * list;
	
    list = gen_linked_list_1(N);

    int p=N/5;

    int i, temp,k;

    for(i=0; i<N; i+=2)
    {
		int k=(i+i+p)%N;
		swap(list,i,k);
    }

    return list;
}

/*
int main()
{
	int N=21;
    //int N=10000000;
    int* qq=NULL;
	int i;
    // qq=gen_linked_list_1(N);
    // printf("\nhere is the list\n");
    // for(i=0; i<N; i++)
        // printf("%3d ", qq[i]);
    // printf("\n");
    // free(qq);
    qq=gen_linked_list_2(N);
    //printf("\nhere is the new list\n");
	printf("%3d ", N);  //输出链表元素个数
	printf("\n");
    for(i=0; i<N; i++)  //输出链表全部元素
        printf("%3d ", qq[i]);
    printf("\n");
    
    free(qq);

    return 0;

}
*/
