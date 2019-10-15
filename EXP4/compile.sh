#!/bin/bash
mpiicc MPI_Trap.cpp -o Trap
mpiexec -n 32 ./Trap $1
