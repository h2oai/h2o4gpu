#!/usr/bin/env bash

#https://setuptools.readthedocs.io/en/latest/setuptools.html?highlight=single-version-externally-managed#install-command

pushd src/interface_py
${PYTHON} setup.py install --single-version-externally-managed --record=record.txt
popd