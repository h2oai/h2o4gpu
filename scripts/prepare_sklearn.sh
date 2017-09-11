#!/bin/bash
cd scikit-learn
rm -rf build dist

########## DIRECTORIES
paths=`find -type d | grep -v pycache | awk '{ print length($0) " " $0; }' | sort  -n | cut -d ' ' -f 2-`

for fil in $paths
do
    echo $fil
    if [[ "$fil" == *".git"* ]]
    then
        echo "avoid .git"
    else
        newpath=`echo $fil | sed 's/sklearn/h2o4gpu/g' | sed 's/scikit-learn/h2o4gpu/g'`
        echo $fil "->" $newpath
        mv $fil $newpath
    fi
done

########## FILES
#files=`find -type f | grep -v pycache`
files=`find -type f | grep -v pycache | awk '{ print length($0) " " $0; }' | sort  -n | cut -d ' ' -f 2-`

for fil in $files
do
    echo $fil
    if [[ "$fil" == *".git"* ]]
    then
        echo "avoid .git"
    else
        sed -i 's/sklearn/h2o4gpu/g' $fil
        sed -i 's/scikit-learn/h2o4gpu/g' $fil
        # replace names
        sed -i 's/KMeans/KMeans_sklearn/g' $fil
        sed -i 's/Ridge/Ridge_sklearn/g' $fil
        sed -i 's/Lasso/Lasso_sklearn/g' $fil
        sed -i 's/LogisticRegression/LogisticRegression_sklearn/g' $fil
        sed -i 's/LinearRegression/LinearRegression_sklearn/g' $fil
        sed -i 's/sklearn_sklearn/sklearn/g' $fil
    fi
done

