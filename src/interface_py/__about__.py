"""
:copyright: 2017-2018 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
__all__ = [ "__version__", "__build_info__" ]

# Build defaults
build_info = {
    'suffix'    : '+local',
    'build'     : 'dev',
    'commit'    : '',
    'describe'  : '',
    'build_os'  : '',
    'build_machine' : '',
    'build_date' : '',
    'build_user' : '',
    'base_version' : '0.0.0',
    'cuda_version' : '',
    'cuda_nccl' : '',
}

import os
path = os.path.join("h2o4gpu", "BUILD_INFO.txt")
if os.path.exists(path):
    with open(path) as f: exec(f.read(), build_info)

# Exported properties to make them available in __init__.py
__version__ = "{}{}".format(build_info['base_version'], build_info['suffix'])
__build_info__ = build_info