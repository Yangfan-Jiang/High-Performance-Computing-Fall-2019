#include<iostream>
#include<fstream>
#include<cstdlib>

using namespace std;

int main() {
    ofstream fon;
    fon.open("data", ios::binary);
    unsigned long len = 100000000;
                        //4294967295;
    unsigned long b =   52949672940;
    unsigned long a = 1;
    fon.write((char*)&len, 8);
    for (unsigned long i=len;i>0;i--) {
        unsigned long L = (rand() % (b-a+1)) + a;
        //unsigned long L= long(x);
        //cout <<L<< " ";
        fon.write((char*)&L, 8);
    }
    return 0;
}
