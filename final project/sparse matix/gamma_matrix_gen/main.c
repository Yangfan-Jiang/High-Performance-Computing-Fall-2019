#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//#define N 12 // $N$ is total numbers of levels. The sparse matrix generated in this file is a 2^N order square matrix.
//#define Q 12 // $Q$ is a constant for controling which subblocks to be generated.
#define DENSE 0
#define NON_DENSE 1
#define Power(b, b_tilde, n, i, j) ( -n + b*(n-j) + b_tilde*(n-i))
#define max(a, b) ((a>b)? a: b)
#define min(a, b) ((a>b)? b: a)
// A celling function for a real number.
#define ceil_i(a) ((((int)a)<a)? (int)(a)+1 : (int)(a))

#define INDICES_STARTING_AT_0


/* *******************************************************
		Weicai Ye's Works
		Dec. 20th, 2007
    This file contains the implementation of computation
    about the distance of two 1-D basis with their indecies,
    with \mu = 2. This version is very fast.
******************************************************* */


//$
//$ Function Name: 	print_dense_block_to_file
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
void print_dense_block_to_file(
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

int nnz_counting(int N, int Q)
{
        // The variables in truncation strategies.
    double a, b, b_tilde, gamma, tmp1, tmp2, tmp3, truncation_parameter[N+1][N+1];
    // The variables for loop control.
    int i, j, l, bound_k, bound_l, lower_bound_k, upper_bound_k, tmp_parameter1, tmp_parameter2;
    // The variables for printing.
    int dense_location[N+1][N+1], offset_diff, tmp_l, nnz=0;

    /// Initinalize parameters in truncation strategies.
    a = 0.25;
    b = 1.0;
    b_tilde = 0.8;
    gamma = 1.01;



    /// Generate parameters of truncation strategies and identify dense blocks.
    for (i=0;i <=N;i++)
    {
        for (j=0;j<=N ; j++)
        {
            tmp1 = a * pow(2, (int)Power(b, b_tilde, N, i, j));
            tmp2 = gamma * (pow(2, -i) + pow(2, -j));
            tmp3 = max(tmp1, tmp2);
            truncation_parameter[i][j] = min(tmp3, 1.0);
            dense_location[i][j] = (tmp3> 0.99) ? DENSE : NON_DENSE;
            // print the trunction parameters of each blocks
        }
    }

    /// Create the sparsity pattern of matrix and print it out.
    for (i=0;i <=Q;i++)
        for (j=0;j<=i ; j++)
        {

            bound_k = (1<<i) - 1;//pow(2, i)-1;
            bound_l = (1<<j) - 1;//pow(2, j)-1;
            offset_diff = 1<< (i-j);

            // Create dense blocks
            if (dense_location[i][j] == DENSE)
            {

                nnz += (1<<(i)) * (1<<(j));
            }
            if ((dense_location[j][i] == DENSE) && (i !=j) )
            {

                nnz += (1<<(i)) * (1<<(j));
            }

            // Create non-dense blocks.
            if ( (dense_location[i][j] == NON_DENSE) || (dense_location[j][i] == NON_DENSE) )
            {
                /// The distance between two special wavelet basis genernated by contract
                /// mappings, which are indexed by positive integers $k$ and $l$,
                /// is defined by hinge_function($s$ - offset_diff(i,j))*2^{i},
                /// where
                ///          hinge_function(a) := (a>0)? a : 0,
                ///          $s$ := abs( offset_diff*l - k ),
                ///          offset_diff(i,j):= 2^{i-j},
                /// $l$ and $k$ are in Z_{2^i} and Z_{2^j}, respectively, with i>=j.
                /// By the above definition, the computation of distance can be simplified
                /// into the following meanings with given $l$, especilly in finding the
                /// indices range which express basis whose distance between $l$ is less than
                /// a given truncation parameter.

                tmp_parameter1 = ceil_i(
                                    truncation_parameter[i][j]*(1<<i) );
                tmp_parameter2 = ceil_i(
                                    truncation_parameter[j][i]*(1<<i) );

                for (l=0;l<=bound_l; l++)
                {
                    tmp_l = l * offset_diff; //k*2^(i-j)

                    //Determine the indices range of nnz with given $l$
                    lower_bound_k = max(0,
                                        tmp_l - tmp_parameter1 - offset_diff );
                    upper_bound_k = min(bound_k,
                                        tmp_parameter1 + offset_diff + tmp_l);

                    nnz += upper_bound_k - lower_bound_k +1;

                    // If this block is not a diangal one, a symmetric block excists.
                    if (i!=j)
                    {
                        lower_bound_k = max(0,
                                            tmp_l - tmp_parameter2 - offset_diff);
                        upper_bound_k = min(bound_k,
                                            tmp_parameter2 + offset_diff + tmp_l);
                        nnz += upper_bound_k - lower_bound_k +1;
                    }
                }
            }
        }

   return nnz;

}

int matrix_gen(int N, int Q)
{

    // The variables in truncation strategies.
    double a, b, b_tilde, gamma, tmp1, tmp2, tmp3, truncation_parameter[N+1][N+1];
    // The variables for loop control.
    int i, j, k, l, bound_k, bound_l, lower_bound_k, upper_bound_k, tmp_parameter1, tmp_parameter2;
    // The variables for printing.
    int dense_location[N+1][N+1], offset_diff, new_l, tmp_l, nnz=0;
    FILE* matrix_file, *index_file;


    // The file names
    const char* matrix_file_name = "matrix12.mat";
    const char* index_file_name = "index.mtx";

    /// Initinalize parameters in truncation strategies.
    a = 0.25;
    b = 1.0;
    b_tilde = 0.8;
    gamma = 1.01;

    // Open files
    matrix_file = fopen(matrix_file_name, "w+");
    index_file = fopen(index_file_name, "w+");
    if ( (!matrix_file) || (!index_file) )
    {
        printf("Can not open files to write a matrix!\n");
        exit(1);
    }
    else
        // If using the matrix-market format file with Matlab, remove the following file header first.
        fprintf(matrix_file,
                "%%%%MatrixMarket matrix coordinate real general\n%d %d %d\n",
                (1<<(N+1))- 1,
                (1<<(N+1))- 1,
                nnz_counting(N, Q));


    /// Generate parameters of truncation strategies and identify dense blocks.
    for (i=0;i <=N;i++)
    {
        for (j=0;j<=N ; j++)
        {
            tmp1 = a * pow(2, (int)Power(b, b_tilde, N, i, j));
            tmp2 = gamma * (pow(2, -i) + pow(2, -j));
            tmp3 = max(tmp1, tmp2);
            truncation_parameter[i][j] = min(tmp3, 1.0);
            dense_location[i][j] = (tmp3> 0.99) ? DENSE : NON_DENSE;
            // print the trunction parameters of each blocks
            fprintf(index_file, " %1.4f ", truncation_parameter[i][j]  );
        }
        fprintf(index_file, "\n");
    }

    /// Create the sparsity pattern of matrix and print it out.
    for (i=0;i <=Q;i++)
        for (j=0;j<=i ; j++)
        {

            bound_k = (1<<i) - 1;//pow(2, i)-1;
            bound_l = (1<<j) - 1;//pow(2, j)-1;
            offset_diff = 1<< (i-j);

            // Create dense blocks
            if (dense_location[i][j] == DENSE)
            {

                nnz += (1<<(i)) * (1<<(j));
#ifdef INDICES_STARTING_AT_0
                // Matrix indices starting at '0' in C language.
                print_dense_block_to_file(
                        matrix_file, bound_k,
                        ( 1<< (i+1) ) -2,
                        bound_l-1,
                        ( 1<< (j+1) ) -2);
#else
                // Matrix indices starting at '1' in Fortran and MatLab.
                print_dense_block_to_file(
                        matrix_file, bound_k+1,
                        (1<< (i+1)) -1,
                        bound_l+1,
                        (1<< (j+1))-1);
#endif
            }
            if ((dense_location[j][i] == DENSE) && (i !=j) )
            {

                nnz += (1<<(i)) * (1<<(j));
#ifdef INDICES_STARTING_AT_0
                // Matrix indices starting at '0' in C language.
                print_dense_block_to_file(
                    matrix_file,
                    bound_l,
                    ( 1<< (j+1) ) -2,
                    bound_k,
                    ( 1<< (i+1) ) -2);
#else
                // Matrix indices starting at '1' in Fortran and MatLab.
                print_dense_block_to_file(
                    matrix_file,
                    bound_l+1,
                    (1<< (j+1))-1,
                    bound_k+1,
                    (1<< (i+1))-1);
#endif
            }

            // Create non-dense blocks.
            if ( (dense_location[i][j] == NON_DENSE) || (dense_location[j][i] == NON_DENSE) )
            {
                /// The distance between two special wavelet basis genernated by contract
                /// mappings, which are indexed by positive integers $k$ and $l$,
                /// is defined by hinge_function($s$ - offset_diff(i,j))*2^{i},
                /// where
                ///          hinge_function(a) := (a>0)? a : 0,
                ///          $s$ := abs( offset_diff*l - k ),
                ///          offset_diff(i,j):= 2^{i-j},
                /// $l$ and $k$ are in Z_{2^i} and Z_{2^j}, respectively, with i>=j.
                /// By the above definition, the computation of distance can be simplified
                /// into the following meanings with given $l$, especilly in finding the
                /// indices range which express basis whose distance between $l$ is less than
                /// a given truncation parameter.

                tmp_parameter1 = ceil_i(
                                    truncation_parameter[i][j]*(1<<i) );
                tmp_parameter2 = ceil_i(
                                    truncation_parameter[j][i]*(1<<i) );

                for (l=0;l<=bound_l; l++)
                {
                    // @new_l is the index of @l in the vector $x$
                    new_l = l + bound_l;
                    tmp_l = l * offset_diff; //k*2^(i-j)

                    //Determine the indices range of nnz with given $l$
                    lower_bound_k = max(0,
                                        tmp_l - tmp_parameter1 - offset_diff );
                    upper_bound_k = min(bound_k,
                                        tmp_parameter1 + offset_diff + tmp_l);

                    nnz += upper_bound_k - lower_bound_k +1;

                    for (k=lower_bound_k; k<=upper_bound_k; k++)
#ifdef INDICES_STARTING_AT_0
                        // Matrix indices starting at '0' in C language.
                        fprintf(matrix_file,
                                "%d %d %1.1f\n",
                                k+bound_k, new_l, 1.0);
#else
                        // Matrix indices starting at '1' in Fortran and MatLab.
                        fprintf(matrix_file,
                                "%d %d %1.1f\n",
                                k+1+bound_k, new_l+1, 1.0);
#endif
                    // If this block is not a diangal one, a symmetric block excists.
                    if (i!=j)
                    {
                        lower_bound_k = max(0,
                                            tmp_l - tmp_parameter2 - offset_diff);
                        upper_bound_k = min(bound_k,
                                            tmp_parameter2 + offset_diff + tmp_l);
                        nnz += upper_bound_k - lower_bound_k +1;

                        for (k=lower_bound_k;k<=upper_bound_k; k++)
#ifdef INDICES_STARTING_AT_0
                            // Matrix indices starting at '0' in C language.
                            fprintf(matrix_file,
                                    "%d %d %1.1f\n",
                                    new_l, k+bound_k, 1.0);
#else
                            // Matrix indices starting at '1' in Fortran and MatLab.
                            fprintf(matrix_file,
                                    "%d %d %1.1f\n",
                                    new_l+1, k+1+bound_k, 1.0);
#endif
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

    int N=12, Q=12;
    if(argc<2)
    {
        printf("Please input a positive no. as the level of bases to be generated, i.e., 12!\n");
        return -1;
    }
    else
    {
        if(atoi(argv[1])<=0)
        {
            printf("Please input a positive no. as the level of bases to be generated, i.e., 12!\n");
            return -2;
        }

        N=Q=atoi(argv[1]);
    }
    matrix_gen(N, Q);
    return 0;
}
