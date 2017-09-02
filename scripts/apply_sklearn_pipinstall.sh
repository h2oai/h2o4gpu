#!/bin/bash

# apply sklearn
rm -rf sklearn
cd scikit-learn
pip install dist/h2o4gpu-0.20.dev0-cp36-cp36m-linux_x86_64.whl --upgrade --target ../sklearn/
cd ../
