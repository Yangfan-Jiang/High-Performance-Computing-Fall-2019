icpc -o a matrix.c -g -lpthread
icpc -o _numa matrix_numa.c -g -lpthread

sudo perf stat -d ./a
echo "---- NUMA case -----"
sudo perf stat -d ./_numa

echo "------ verify result ------"
file1=result1
file2=result_numa
diff $file1 $file2 > /dev/null
if [ $? == 0 ]
then
    echo "verify successfully!"
else
    echo "!! wrong answer !!"
fi
