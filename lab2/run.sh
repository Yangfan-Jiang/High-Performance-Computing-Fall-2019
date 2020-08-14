icpc -o a matrix.c -g -lpthread
icpc -o b false_sharing.c -g -lpthread

sudo perf stat -d ./a
echo "---- Cacahe false sharing case -----"
sudo perf stat -d ./b

echo "------ verify result ------"
file1=result1
file2=result2
diff $file1 $file2 > /dev/null
if [ $? == 0 ]
then
    echo "verify successfully!"
else
    echo "!! wrong answer !!"
fi
