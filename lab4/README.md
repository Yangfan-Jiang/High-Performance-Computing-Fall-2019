# MPI实验
- MPI_Trap.cpp为梯形积分实验，对应的执行脚本为compile.sh，执行时需要带一个参数0/1选择通信方式
- 常规的数值积分和瑕积分实现都在MPI_Trap.cpp文件中，根据需要更改41/42行注释即可
- MPI_PSRS.cpp为正则采样排序的MPI实现，对应的执行脚本为run_psrs.sh