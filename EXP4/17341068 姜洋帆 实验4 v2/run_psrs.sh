#!/bin/bash
mpic++ MPI_PSRS.cpp -o PSRS
qsub test.pbs

