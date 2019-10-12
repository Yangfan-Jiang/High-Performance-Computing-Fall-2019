#rm jyfTest.*
g++ matrix.cpp -o a -lpthread -std=c++11
qsub test.pbs
