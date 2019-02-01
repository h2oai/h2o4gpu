#!/bin/bash

# apply sklearn
rm -rf sklearn
cd scikit-learn
file=`ls dist/h2o4gpu*.whl`

python -m pip install $file --upgrade --target ../sklearn/
cd ../
