# -*- encoding: utf-8 -*-
"""
:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
from functools import wraps

def print_name(func):
    @wraps(func)
    def full(*args, **kwargs):
        print("**-> {}".format(func.__name__))
        return func(*args, **kwargs)
    return full
