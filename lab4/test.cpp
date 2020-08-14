#include<iostream>
#include<cmath>

using namespace std;

double a = 10;
double b = 50;

void IntervalCoor(int n, int internal_num) {
    double intern_len;
    double coor;

    int n_inv = n-internal_num;
    int intern_total = floor(log2(n)) + 1;
    int x = floor(log2(n_inv));
    
    double block_size = (b-a) / intern_total;
    
    if(internal_num == n) {
        intern_len = block_size;
        coor = b;
    } else {
        intern_len = (b-a) / intern_total / pow(2, x);
        coor = b - block_size * (x+1) - (n_inv%int(pow(2, x))) * intern_len;
    }
}

int main() {
    for (int i=1; i <= 32; i++) {
        IntervalCoor(32, i);
    }
}