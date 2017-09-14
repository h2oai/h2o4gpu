#!/bin/bash
rm -rf scikit-learn
git submodule init
git submodule update
cd scikit-learn

########## DIRECTORIES and FILENAMES
echo "Renaming paths and files"
find ./ -execdir rename -v 's/sklearn/h2o4gpu/' '{}' \+
find ./ -execdir rename -v 's/scikit-learn/h2o4gpu/' '{}' \+

########## FILE contents
#files=`find -type f | grep -v pycache`
files=`find -type f | grep -v pycache | awk '{ print length($0) " " $0; }' | sort  -n | cut -d ' ' -f 2-`

for fil in $files
do
    echo "Edit contents of $fil"
    if [[ "$fil" == *".git"* ]]
    then
        #echo "skip .git"
        true
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
        sed -i "s/from \.\. import get_config as _get_config/import os\n_ASSUME_FINITE = bool(os.environ.get('SKLEARN_ASSUME_FINITE', False))\ndef _get_config\(\):\n    return \{'assume_finite': _ASSUME_FINITE\}/g" $fil
    fi
done

