#!/bin/bash

# handle base __init__.py file appending
rm -rf src/interface_py/h2o4gpu/__init__.py
cat sklearn/h2o4gpu/__init__.py | sed 's/__version__.*//g' >> src/interface_py/h2o4gpu/__init__.py.2

cat >> src/interface_py/h2o4gpu/__init__.py <<EOL
"""
:copyright: 2017-2019 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
# stub value	# Skip pylint b / c this is automatically concatenated at compile time
__version__ = "0.0.0.0000"
EOL

cat src/interface_py/build_info.txt src/interface_py/h2o4gpu/__init__.py.2  src/interface_py/h2o4gpu/__init__.base.py >> src/interface_py/h2o4gpu/__init__.py
rm -rf src/interface_py/h2o4gpu/__init__.py.2
