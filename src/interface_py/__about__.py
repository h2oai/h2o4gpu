#!/usr/bin/env python
# Copyright 2017 H2O.ai; Proprietary License;  -*- encoding: utf-8 -*-
import pathlib

__all__ = [ "__version__", "__build_info__" ]

# Build defaults
build_info = {
    'suffix'    : '',
    'build'     : 'dev',
    'commit'    : '',
    'describe'  : '',
    'build_os'  : '',
    'build_machine' : '',
    'build_date' : '',
    'build_user' : '',
    'base_version' : '0.0.0'
}

import pkg_resources
if pathlib.Path('h2o4gpu/BUILD_INFO.txt').is_file():
    exec(pkg_resources.resource_string('h2o4gpu', 'BUILD_INFO.txt'), build_info)

# Exported properties
__version__ = "{}{}".format(build_info['base_version'], build_info['suffix'])
__build_info__ = build_info