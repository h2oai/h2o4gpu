#!/usr/bin/env bash

#https://setuptools.readthedocs.io/en/latest/setuptools.html?highlight=single-version-externally-managed#install-command
make deps_fetch
# Do we need to install R dependencies as part of make deps_install
# bash scripts/install_r_deps.sh                
make xgboost
make install_xgboost
make fullinstall-lightgbm
make libsklearn
make cpp
make apply-sklearn_simple
make py
make install_py

pushd src/interface_py
${PYTHON} setup.py install --single-version-externally-managed --record=record.txt
popd