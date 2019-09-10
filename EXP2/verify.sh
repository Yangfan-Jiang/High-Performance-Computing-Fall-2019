file1=result1
file2=result2
diff $file1 $file2 > /dev/null
if [ $? == 0 ] 
then
    echo "same"
else
    echo "different"
fi
