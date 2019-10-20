#include<iostream>
#include<mpi.h>
#include<cmath>

using namespace std;


double Trap(double, double, int, double);
void IntervalCoor(double, double, int, int, double*, double*);
double ImproperInt(double, double, int, int, int);
double f(double);

const int c = 1;
const int d = 1;

int main(int argc, char* argv[])
{
    int my_rank, comm_sz, n = 4096, local_n;
    double a = 0.0, b = 1.0, h, local_a, local_b;
    double local_int, total_int;
    int source;
    const int k = sqrt(n);
    
    double start, end;
    
    int mode = atoi(argv[1]);
    
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    
    h = (b-a) / n;
    local_n = n / comm_sz;
    
    local_a = a + my_rank*local_n*h;
    local_b = local_a + local_n*h;
    
    //local_int = Trap(local_a, local_b, local_n, h);
    local_int = ImproperInt(a, b, my_rank, local_n, n);
    
    // point to point
    if(mode == 0) {   
        if (my_rank != 0) {
            MPI_Send(&local_int, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        } else {
            total_int = local_int;
            for (source = 1; source < comm_sz; source++) {
                MPI_Recv(&local_int, 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                total_int += local_int;
            }
        }
    } // Reduce
    else if(mode == 1) {
        MPI_Reduce(&local_int, &total_int, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    
    MPI_Finalize();
    
    if (my_rank == 0) {
        cout << "elapsed: "<< (end - start)*1000 << "ms" << endl;
        cout << "a = " << a << ", b = " << b << ", processes number: " << comm_sz << endl;
        cout << "result: " << total_int << endl;
    }
    
    return 0;
}


/* Numerical Integration */
double Trap(double a, double b, int local_n, double h) {
    double approx;
    double sum = 0;
    
    sum = h*(f(a) + f(b))/2.0;
    for (int i = 1; i <= local_n-1; i++) {
        double x1 = a + i*h;
        double x2 = a + (i+1)*h;
        
        approx = (f(x1) + f(x2)) / 2.0;
        sum += approx*h;
    }
    return sum;
}

/* Improper Integral */
double ImproperInt(double a, double b, int rank, int local_n, int n) {
    double approx;
    double sum = 0;
    double intern_len;
    double coor;
    int start_intern = rank*local_n+1;
    
    IntervalCoor(a, b, n, start_intern, &intern_len, &coor);
    sum += intern_len*(f(coor))/2.0;
    for (int i = 1; i <= local_n-1; i++) {
        start_intern += 1;
        IntervalCoor(a, b, n, start_intern, &intern_len, &coor);
        approx = f(coor);
        sum += approx*intern_len;
    }

    return sum;
}

/* get the length of intern and start coordinate */
void IntervalCoor(double a, double b, int n, int internal_num, double* intern_len, double* coor) {
    int n_inv = n-internal_num;
    int intern_total = floor(log2(n)) + 1;
    int x = floor(log2(n_inv));
    
    double block_size = (b-a)/intern_total*1.0;
    
    if(internal_num == n) {
        *intern_len = block_size;
        *coor = b;
    } else {
        *intern_len = (b-a) / intern_total / pow(2, x);
        *coor = b - block_size * (x+1) - (n_inv%int(pow(2, x))) * (*intern_len);
    }
}

double f(double x) {
    return exp(c*x)/sqrt(1-exp(-d*x));
    //return exp(x);
    //return 1.0/(x*x);
}


