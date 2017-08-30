cd src/interface_py/h2o4gpu/
# now register any override of existing classes
# e.g. from grep -R 'from \.' * |grep KMeans
infile="register.clean.dat"
grep -v "#" register.dat > $infile

#IFS=' '
while read skfile initfile classorig classnew command
do

    #classorig=`sed -e 's/\"//g' <<< $classorig`

    #temp=${classnew%\"}
    #temp=${temp#\"}
    #classnew=`echo $temp`

    echo $skfile
    echo $initfile
    echo $classorig
    echo $classnew
    echo $command

    if [ 1 -eq 1 ]
       then
           # rename class
           echo "1"
           sed -i "s/class $classorig(/class $classnew(/g" $skfile
           echo "2"
           sed -i "s/($classorig)/($classnew)/g" $skfile
           echo "3"
           sed -i "s/sklearn_sklearn/sklearn/g" $skfile
           echo "4"
           sed -i "s/\ $classorig\([^[a-z]\)/ $classnew\1/g" $skfile
           echo "5"
           sed -i "s/sklearn_sklearn/sklearn/g" $skfile

           # rename class in init file
           #sed -i "s/\([^[a-z]\)$classorig\([^[a-z]\)/\1$classnew\2/g" $initfile
           echo "6"
           sed -i "s/\ $classorig\([^[a-z]\)/ $classnew\1/g" $initfile
           echo "7"
           sed -i "s/sklearn_sklearn/sklearn/g" $initfile

           # append our class
           echo "" >> $skfile
           command=`sed -e 's/^"//' -e 's/"$//' <<< "$command"`
           echo "$command" >> $initfile
    fi
    
done < $infile

rm -rf $infile
