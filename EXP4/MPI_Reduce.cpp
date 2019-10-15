#include<iostream>
#include<mpi.h>
#include<cmath>

using namespace std;


double Trap(double, double, int, double);
double f(double);

int main()
{
    int my_rank, comm_sz, n = 1024, local_n;
    double a = 1.0, b = 3.0, h, local_a, local_b;
    double local_int, total_int;
    int source;
    
    double start, end;
    
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    
    h = (b-a) / n;
    local_n = n / comm_sz;
    
    local_a = a + my_rank*local_n*h;
    local_b = local_a + local_n*h;
    local_int = Trap(local_a, local_b, local_n, h);
    
    /*
    if (my_rank != 0) {
        MPI_Send(&local_int, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    } else {
        total_int = local_int;
        for (source = 1; source < comm_sz; source++) {
            MPI_Recv(&local_int, 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_int += local_int;
        }
    }
    */
    
    MPI_Reduce(&local_int, &total_int, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD); /* IMPORTANT */
    end = MPI_Wtime();
    
    MPI_Finalize();
    
    if (my_rank == 0) {
        cout << "a = " << a << ", b = " << b << ", processes number: " << comm_sz << endl;
        cout << "result: " << total_int << endl;
    }
    
    return 0;
}


/* Numerical Integration */
double Trap(double a, double b, int n, double h) {
    double approx;
    double sum = 0;
    
    sum = h*(f(a) + f(b))/2.0;
    for (int i = 1; i <= n-1; i++) {
        double x1 = a + i*h;
        double x2 = a + (i+1)*h;
        
        approx = (f(x1) + f(x2)) / 2.0;
        sum += approx*h;
    }
    return sum;
}

double f(double x) {
    return 1.0/(x*x);
}


