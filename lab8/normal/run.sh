# mv cuda_mat_multi.c cuda_mat_multi.cu
nvcc cuda_mat_multi.cu -o a
qsub cuda.pbs
