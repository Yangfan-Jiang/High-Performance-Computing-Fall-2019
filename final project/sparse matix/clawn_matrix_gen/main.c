#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//#define N 8
//#define Q 4
#define DENSE 0
#define NON_DENSE 1
#define Power(b, b_tilde, n, i, j) ( -n + b*(n-j) + b_tilde*(n-i))
#define max(a, b) ((a>b)? a: b)
#define min(a, b) ((a>b)? b: a)
#define hinge_function(a) ((a>0)? a: 0)
// A absolution function for integals
#define abs_i(a) ((a>0)? a : -(a))
#define INDICES_STARTING_AT_0


/* *******************************************************
		Weicai Ye's Works
		Dec. 9th, 2007
    This file contains the implementation of computation
    about the distance of two 1-D basis with their indecies.
******************************************************* */


//$
//$ Function Name: 	print_dense_matrix_to_file
//$
//$ Description:
//$   This method prints a dense matrix into a given file.
//$
//$ Parameters:
//$     file_handle - The given destination file handle.
//$     row_start, row_end, column_start, column_end
//$                 - The indices express the location
//$         of the matrix to be printed, whose rows start
//$         at row_start and end at row_end, and whose columns
//$         start at column_start and end at column_end,
//$ Return:
//$     None.
//$

void print_dense_matrix_to_file(
    FILE* file_handle,
    int row_start,
    int row_end,
    int column_start,
    int column_end)
{
    int k, l;
    for (k=row_start; k<=row_end; k++)
        for (l=column_start; l<=column_end; l++)
            fprintf(file_handle, "%d %d %1.1f\n", k, l, 1.0);
}


int counting_nnz(int N, int Q)
{
    double a, b, b_tilde, gamma, tmp1, tmp2, tmp3, truncation_parameter[N+1][N+1], tmp_parameter1, tmp_parameter2;
    int i=0, j=0, k=0, l=0, bound_k, bound_l, dense_location[N+1][N+1], offset_diff, tmp_l, nnz=0, flag1, flag2;

    a = 0.25;
    // a = 0.125;
    b = 1.0;
    // b = 0.9;
    b_tilde = 0.8;
    //b_tilde = 0.6;
    gamma = 1.01;
    //gamma = 2.01;

    /// Generate distribution of truncation strategies and dense blocks, also the parameters
    for (i=0;i <=N;i++)
    {
        for (j=0;j<=N ; j++)
        {
            tmp1 = a * pow(2, (int)Power(b, b_tilde, N, i, j));
            tmp2 = gamma * (pow(2, -i) + pow(2, -j));
            tmp3 = max(tmp1, tmp2);
            truncation_parameter[i][j] = min(tmp3, 1.0);
            dense_location[i][j] = (tmp3> 0.99) ? DENSE : NON_DENSE;
        }

    }

    for (i=0;i <=Q;i++)
        for (j=0;j<=i ; j++)
        {

            bound_k = (1<<i) - 1;//pow(2, i)-1;
            bound_l = (1<<j) - 1;//pow(2, j)-1;
            offset_diff = 1<< (i-j);

            // Dense blocks
            if ( (dense_location[i][j] == DENSE) || ((dense_location[j][i] == DENSE)) )
            {
                if (dense_location[i][j] == DENSE)
                {

                    nnz += (1<<(i)) * (1<<(j));
                }
                if ((dense_location[j][i] == DENSE) && (i !=j) )
                {

                    nnz += (1<<(i)) * (1<<(j));
                }

            }
            // Non-dense blocks
            if ( (dense_location[i][j] == NON_DENSE) || (dense_location[j][i] == NON_DENSE) )
            {
                tmp_parameter1 = truncation_parameter[i][j]*(1<<i);
                tmp_parameter2 = truncation_parameter[j][i]*(1<<i);

                //printf("%f %f_ \n", tmp_parameter1, truncation_parameter[i][j]);

                for (l=0;l<=bound_l; l++)
                {
                    tmp_l = l * offset_diff; //k*2^(i-j)

                    for (k=0;k<=bound_k; k++)
                    {
                        tmp1 = hinge_function(abs_i(tmp_l - k) -offset_diff ) ;
                        flag1 = (tmp1 > tmp_parameter1)? 0: 1;
                        flag2 = (tmp1 > tmp_parameter2)? 0: 1;

                        if (flag1 || flag2 )
                        {
                            if ( flag1==1 )
                            {
                                nnz++;
                            }
                            if ((flag2==1) && (i!=j) )
                            {
                                nnz++;
                            }
                        }
                    }
                }
            }
        }

    return nnz;
}


int matrix_gen(int N, int Q)
{
    double a, b, b_tilde, gamma, tmp1, tmp2, tmp3, truncation_parameter[N+1][N+1], tmp_parameter1, tmp_parameter2;
    int i=0, j=0, k=0, l=0, bound_k, bound_l, line_counter=1,  dense_location[N+1][N+1], offset_diff,  new_l, tmp_l, nnz=0, flag1, flag2;
    FILE* matrix_file, *index_file;
    const char* matrix_file_name = "matrix.mat";
    const char* index_file_name = "index.mtx";
    a = 0.25;
    // a = 0.125;
    b = 1.0;
    // b = 0.9;
    b_tilde = 0.8;
    //b_tilde = 0.6;
    gamma = 1.01;
    //gamma = 2.01;
    matrix_file = fopen(matrix_file_name, "w+");
    index_file = fopen(index_file_name, "w+");
    if ( (!matrix_file) || (!index_file) )
    {
        printf("Can not open files to write a matrix!\n");
        exit(1);
    }
    else
    {
        // If using the matrix-market format file with Matlab, remove the following file header first.
        fprintf(matrix_file, "%%%%MatrixMarket matrix coordinate real general\n%d %d %d\n", (1<<(N+1))- 1, (1<<(N+1))- 1,  counting_nnz(N, Q));
        //740794);
        line_counter += 2;
        // print the record count of index file
        fprintf(index_file, "%d\n", (N+1)*(N*1));
    }


    /// Generate distribution of truncation strategies and dense blocks, also the parameters
    for (i=0;i <=N;i++)
    {
        for (j=0;j<=N ; j++)
        {
            tmp1 = a * pow(2, (int)Power(b, b_tilde, N, i, j));
            tmp2 = gamma * (pow(2, -i) + pow(2, -j));
            tmp3 = max(tmp1, tmp2);
            truncation_parameter[i][j] = min(tmp3, 1.0);
            dense_location[i][j] = (tmp3> 0.99) ? DENSE : NON_DENSE;
            fprintf(index_file, " %1.4f ", truncation_parameter[i][j]  );
        }
        fprintf(index_file, "\n");
    }

    for (i=0;i <=Q;i++)
        for (j=0;j<=i ; j++)
        {

            bound_k = (1<<i) - 1;//pow(2, i)-1;
            bound_l = (1<<j) - 1;//pow(2, j)-1;
            offset_diff = 1<< (i-j);

            // Dense blocks
            if ( (dense_location[i][j] == DENSE) || ((dense_location[j][i] == DENSE)) )
            {
                if (dense_location[i][j] == DENSE)
                {

                    nnz += (1<<(i)) * (1<<(j));
#ifdef INDICES_STARTING_AT_0
                    // Matrix indices starting at '0' in C language.
                    //print_dense_matrix_to_file(matrix_file, bound_k, ( 1<< (i+1) ) -2, bound_l-1, ( 1<< (j+1) ) -2);
                    print_dense_matrix_to_file(matrix_file, bound_k, ( 1<< (i+1) ) -2, bound_l, ( 1<< (j+1) ) -2);
#else
                    // Matrix indices starting at '1' in Fortran and MatLab.
                    print_dense_matrix_to_file(matrix_file, bound_k+1, (1<< (i+1)) -1, bound_l+1, (1<< (j+1))-1);
#endif
                }
                if ((dense_location[j][i] == DENSE) && (i !=j) )
                {

                    nnz += (1<<(i)) * (1<<(j));
#ifdef INDICES_STARTING_AT_0
                    // Matrix indices starting at '0' in C language.
                    print_dense_matrix_to_file(matrix_file, bound_l, ( 1<< (j+1) ) -2, bound_k, ( 1<< (i+1) ) -2);
#else
                    // Matrix indices starting at '1' in Fortran and MatLab.
                    print_dense_matrix_to_file(matrix_file, bound_l+1, (1<< (j+1))-1, bound_k+1, (1<< (i+1))-1);
#endif
                }

            }
            // Non-dense blocks
            if ( (dense_location[i][j] == NON_DENSE) || (dense_location[j][i] == NON_DENSE) )
            {
                tmp_parameter1 = truncation_parameter[i][j]*(1<<i);
                tmp_parameter2 = truncation_parameter[j][i]*(1<<i);

                //printf("%f %f_ \n", tmp_parameter1, truncation_parameter[i][j]);

                for (l=0;l<=bound_l; l++)
                {

                    new_l = l + bound_l;
                    tmp_l = l * offset_diff; //k*2^(i-j)

                    for (k=0;k<=bound_k; k++)
                    {
                        tmp1 = hinge_function(abs_i(tmp_l - k) -offset_diff ) ;
                        flag1 = (tmp1 > tmp_parameter1)? 0: 1;
                        flag2 = (tmp1 > tmp_parameter2)? 0: 1;

                        fprintf(index_file, "%d\n", hinge_function(abs_i( tmp_l - k ) -offset_diff )   );


                        if (flag1 || flag2 )
                        {
                            if ( flag1==1 )
                            {

#ifdef INDICES_STARTING_AT_0
                                // Matrix indices starting at '0' in C language.
                                fprintf(matrix_file, "%d %d %1.1f\n", k+bound_k, new_l, 1.0);
#else
                                // Matrix indices starting at '1' in Fortran and MatLab.
                                fprintf(matrix_file, "%d %d %1.1f\n", k+1+bound_k, new_l+1, 1.0);
#endif

                                nnz++;
                            }
                            if ((flag2==1) && (i!=j) )
                            {

#ifdef INDICES_STARTING_AT_0
                                // Matrix indices starting at '0' in C language.
                                fprintf(matrix_file, "%d %d %1.1f\n", new_l, k+bound_k, 1.0);
#else
                                // Matrix indices starting at '1' in Fortran and MatLab.
                                fprintf(matrix_file, "%d %d %1.1f\n", new_l+1, k+1+bound_k, 1.0);
#endif
                                nnz++;
                            }
                        }
                    }
                }
            }
        }


    if ( fclose(matrix_file) || fclose(index_file) )
    {
        printf("Can not close files to write a matrix!\n");
        exit(1);
    }

    // Don't forget to add the total nnz to the matrix file.
    printf("Total nnz is: %d\n", nnz);


    return 0;
}

//$
//$ Function Name: 	main
//$
//$ Description:
//$   This main function to generate a sparse matrix.
//$
//$ Parameters:
//$     None.
//$
//$ Return:
//$     0 - Successful.
//$     1 - Failed.
//$
int main(int argc,char *argv[ ])
{

//    int N=8, Q=4;
    int N=12, Q=12;
    if(argc<3)
    {
        printf("Please input two positive no.s as the levels of bases to be generated, i.e., 12, 8!\n");
        return -1;
    }
    else
    {
        if( (atoi(argv[1])<=0) || (atoi(argv[2])<=0) )
        {
            printf("Please input a positive no. as the level of bases to be generated, i.e., 12!\n");
            return -2;
        }

        N=atoi(argv[1]);
        Q=atoi(argv[2]);
    }
    matrix_gen(N, Q);
    return 0;
}

