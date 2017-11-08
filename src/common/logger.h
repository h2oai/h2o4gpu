/*!
 * Copyright 2017 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#pragma once

#include <stdio.h>
#include <stdarg.h>

#define H2O4GPU_LOG_NOTHING    0   // Fatals are errors terminating the program immediately
#define H2O4GPU_LOG_FATAL    100   // Fatals are errors terminating the program immediately
#define H2O4GPU_LOG_ERROR    200   // Errors are when the program may not exit
#define H2O4GPU_LOG_INFO     300   // Info
#define H2O4GPU_LOG_WARN     400   // Warns about unwanted, but not dangerous, state/behaviour
#define H2O4GPU_LOG_DEBUG    500   // Most basic debug information
#define H2O4GPU_LOG_VERBOSE  600   // Everything possible

#define log_fatal(desired_level, ...) log(desired_level, H2O4GPU_LOG_FATAL, __FILE__, __LINE__, __VA_ARGS__)
#define log_error(desired_level, ...) log(desired_level, H2O4GPU_LOG_ERROR, __FILE__, __LINE__, __VA_ARGS__)
#define log_info(desired_level, ...)  log(desired_level, H2O4GPU_LOG_INFO,  __FILE__, __LINE__, __VA_ARGS__)
#define log_warn(desired_level, ...)  log(desired_level, H2O4GPU_LOG_WARN,  __FILE__, __LINE__, __VA_ARGS__)
#define log_debug(desired_level, ...) log(desired_level, H2O4GPU_LOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define log_verbose(desired_level, ...) log(desired_level, H2O4GPULOG_VERBOSE, __FILE__, __LINE__, __VA_ARGS__)

void log(int desired_level, int level, const char *file, int line, const char *fmt, ...);