#!/bin/bash

cd src/interface_py/h2o4gpu

# PATHS
sklearnpaths=`find ../../../sklearn/h2o4gpu -type d  | tail -n +2 | grep -v pycache | awk '{ print length($0) " " $0; }' | sort  -n | cut -d ' ' -f 2-`

for fil in $sklearnpaths
do

    file=`basename $fil`
    path=`dirname $fil | sed 's/\.\.\/\.\.\/\.\.\/sklearn\/h2o4gpu//g' | sed 's/^\///g' | sed 's/^/\.\//g'`
    newfile=${path}/$file
    echo $fil `dirname $fil` $file $path $newfile
    
    echo "mkdir: " $newfile
    mkdir -p $newfile
    #rm -rf $newfile

done

if [ 1 -eq 1 ]
   then
# FILES
sklearnfiles=`find ../../../sklearn/h2o4gpu -type f | grep -v pycache`

for fil in $sklearnfiles
do

    file=`basename $fil`
    path=`dirname $fil | sed 's/\.\.\/\.\.\/\.\.\/sklearn\/h2o4gpu//g' | sed 's/^\///g' | sed 's/^/\.\//g'`
    newfile=${path}/$file

    echo $fil "->" $newfile

    #rm -rf $path
    #git rm -rf $newfile
    ln -sfr $fil $newfile

done
fi
