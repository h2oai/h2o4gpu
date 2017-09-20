#- * - encoding : utf - 8 - * -
"""
:mod:`h2o.exceptions` -- all exceptions classes in h2o module.

All H2O4GPU exceptions derive from :class:`H2O4GPUError`.

:copyright: 2017 H2O.ai, Inc.
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
from __future__ import absolute_import, division, print_function, \
    unicode_literals

__all__ = ("H2O4GPUValueError", "H2O4GPUTypeError")


class H2O4GPUError(Exception):
    """Base class for all H2O4GPU exceptions."""


class H2O4GPUSoftError(H2O4GPUError):
    """Base class for exceptions that trigger "soft" exception handling hook."""


#-- -- -- -- -- --
#H2O4GPUValueError
#-- -- -- -- -- --


class H2O4GPUValueError(H2O4GPUSoftError, ValueError):
    """Error indicating that wrong parameter value was passed to a function."""

    def __init__(self, message, var_name=None, skip_frames=0):
        """Create an H2O4GPUValueError exception object."""
        super(H2O4GPUValueError, self).__init__(message)
        self.var_name = var_name
        self.skip_frames = skip_frames


#-- -- -- -- -- --
#H2O4GPUTypeError
#-- -- -- -- -- --


class H2O4GPUTypeError(H2O4GPUSoftError, TypeError):
    """
    Error indicating that the user passed a parameter of wrong type.

    This error will trigger "soft" exception handling, in the sense that the
    stack trace will be much more compact than usual.
    """

    def __init__(self,
                 var_name=None,
                 var_value=None,
                 var_type_name=None,
                 exp_type_name=None,
                 message=None,
                 skip_frames=0):
        """
        Create an H2O4GPUTypeError exception object.

        :param message: error message that will be shown to the user. If not
            given, this message will be constructed from ``var_name``,
            ``var_value``, etc.
        :param var_name: name of the variable whose type is wrong (can be used
            for highlighting etc).
        :param var_value: the value of the variable.
        :param var_type_name: the name of the variable's actual type.
        :param exp_type_name: the name of the variable's expected type.
        :param skip_frames: how many auxiliary function calls have been made
            since the moment of the exception. This many local frames will be
            skipped in the output of the exception message. For example if you
            want to check a variables type, and call a helper
            function ``assert_is_type()`` to do that job for you, then
            ``skip_frames`` should be 1 (thus making the call to
            ``assert_is_type`` invisible).
        """
        super(H2O4GPUTypeError, self).__init__(message)
        self._var_name = var_name
        self._var_value = var_value
        self._var_type_name = var_type_name or str(type(var_value))
        self._exp_type_name = exp_type_name
        self._message = message
        self._skip_frames = skip_frames

    def __str__(self):
        """Used when printing out the exception message."""
        if self._message:
            return self._message


#Otherwise construct the message
        var = self._var_name
        val = self._var_value
        atn = self._var_type_name
        etn = self._exp_type_name or ""
        article = "an" if etn.lstrip("?")[0] in "aioeH" else "a"
        return "Argument `{var}` should be {an} {expected_type}, " \
               "got {actual_type} {value}". \
            format(var=var, an=article, expected_type=etn, actual_type=atn,
                   value=val)

    @property
    def var_name(self):
        """Variable name."""
        return self._var_name

    @property
    def skip_frames(self):
        """Number of local frames to skip when printing our the stacktrace."""
        return self._skip_frames
