#!/usr/bin/env bash

pushd src/interface_py
${PYTHON} setup.py install --single-version-externally-managed --record=record.txt
popd