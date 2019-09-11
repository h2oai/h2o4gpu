#!/bin/bash

# apply sklearn
rm -rf sklearn
cd scikit-learn
file=`ls dist/h2o4gpu*.whl`

pip install $file --upgrade --constraint ../src/interface_py/requirements_buildonly.txt --target ../sklearn/
cd ../
