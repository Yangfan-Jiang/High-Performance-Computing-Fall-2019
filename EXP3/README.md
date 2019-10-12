# 高性能实验三
- 代码在matrix.cpp和matrix_busy.cpp中，其中matrix_busy.cpp为忙等待实现同步
- test.pbs为作业脚本，将作业提交至集群
- compile.sh为执行脚本，会编译代码并执行pbs脚本-(默认为编译matrix.cpp代码)
- 矩阵输入规模以及矩阵划分策略可以通过调整代码8-14行的宏定义来修改