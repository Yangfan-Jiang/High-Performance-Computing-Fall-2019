#!/bin/bash
mpic++ -o matrix -fopenmp matrix.cpp -O0
qsub sub1.pbs


