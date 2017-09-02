#!/bin/bash

# handle base __init__.py file appending
rm -rf src/interface_py/h2o4gpu/__init__.py
cat sklearn/h2o4gpu/__init__.py | sed 's/__version__.*//g' >> src/interface_py/h2o4gpu/__init__.py.2

cat src/interface_py/h2o4gpu/__init__.base.py src/interface_py/h2o4gpu/__init__.py.2 > src/interface_py/h2o4gpu/__init__.py
rm -rf src/interface_py/h2o4gpu/__init__.py.2
