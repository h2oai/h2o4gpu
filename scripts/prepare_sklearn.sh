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
        sed -i 's/\([^_a-zA-Z0-9]\?\)KMeans\([^_a-zA-Z0-9]\?\)/\1KMeansSklearn\2/g' $fil
        sed -i 's/\([^_a-zA-Z0-9]\?\)Ridge\([^_a-zA-Z0-9]\?\)/\1RidgeSklearn\2/g' $fil
        sed -i 's/\([^_a-zA-Z0-9]\?\)Lasso\([^_a-zA-Z0-9]\?\)/\1LassoSklearn\2/g' $fil
        sed -i 's/\([^_a-zA-Z0-9]\?\)LogisticRegression\([^_a-zA-Z0-9]\?\)/\1LogisticRegressionSklearn\2/g' $fil
        sed -i 's/\([^_a-zA-Z0-9]\?\)LinearRegression\([^_a-zA-Z0-9]\?\)/\1LinearRegressionSklearn\2/g' $fil 
        sed -i 's/\([^_a-zA-Z0-9]\?\)ElasticNet\([^_a-zA-Z0-9]\?\)/\1ElasticNetSklearn\2/g' $fil	
        sed -i 's/\([^_a-zA-Z0-9]\?\)GradientBoostingRegressor\([^_a-zA-Z0-9]\?\)/\1GradientBoostingRegressorSklearn\2/g' $fil
        sed -i 's/\([^_a-zA-Z0-9]\?\)GradientBoostingClassifier\([^_a-zA-Z0-9]\?\)/\1GradientBoostingClassifierSklearn\2/g' $fil
        sed -i 's/\([^_a-zA-Z0-9]\?\)RandomForestRegressor\([^_a-zA-Z0-9]\?\)/\1RandomForestRegressorSklearn\2/g' $fil
        sed -i 's/\([^_a-zA-Z0-9]\?\)RandomForestClassifier\([^_a-zA-Z0-9]\?\)/\1RandomForestClassifierSklearn\2/g' $fil
        sed -i 's/\([^_a-zA-Z0-9]\?\)TruncatedSVD\([^_a-zA-Z0-9]\?\)/\1TruncatedSVDSklearn\2/g' $fil
        sed -i 's/\([^_a-zA-Z0-9]\?\)PCA\([^_a-zA-Z0-9]\?\)/\1PCASklearn\2/g' $fil
	# avoid duplicate conversions
        sed -i 's/Sklearn_Sklearn/Sklearn/g' $fil
        # other replacements
        sed -i "s/from \.\. import get_config as _get_config/import os\n_ASSUME_FINITE = bool(os.environ.get('SKLEARN_ASSUME_FINITE', False))\ndef _get_config\(\):\n    return \{'assume_finite': _ASSUME_FINITE\}/g" $fil
    fi
done
cd ..

# inject h2o4gpu into scikit-learn
echo "import h2o4gpu.solvers.kmeans"               >> scikit-learn/h2o4gpu/cluster/__init__.py
echo "import h2o4gpu.solvers.kmeans"               >> scikit-learn/h2o4gpu/cluster/__init__.py
echo "import h2o4gpu.solvers.ridge"                >> scikit-learn/h2o4gpu/linear_model/__init__.py
echo "import h2o4gpu.solvers.lasso"                >> scikit-learn/h2o4gpu/linear_model/__init__.py
echo "import h2o4gpu.solvers.logistic"             >> scikit-learn/h2o4gpu/linear_model/__init__.py
echo "import h2o4gpu.solvers.linear_regression"    >> scikit-learn/h2o4gpu/linear_model/__init__.py
echo "import h2o4gpu.solvers.elastic_net"          >> scikit-learn/h2o4gpu/linear_model/__init__.py
