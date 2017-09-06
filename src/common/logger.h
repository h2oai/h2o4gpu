/*!
 * Copyright (c) 2017 H2O.ai
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#pragma once

#define H2O4GPU_LOG_NOTHING    0   // Fatals are errors terminating the program immediately
#define H2O4GPU_LOG_FATAL    100   // Fatals are errors terminating the program immediately
#define H2O4GPU_LOG_ERROR    200   // Errors are when the program may not exit
#define H2O4GPU_LOG_INFO     300   // Info
#define H2O4GPU_LOG_WARN     400   // Warns about unwanted, but not dangerous, state/behaviour
#define H2O4GPU_LOG_DEBUG    500   // Most basic debug information
#define H2O4GPU_LOG_VERBOSE  600   // Everything possible

void log_fatal(const int verbosity, const char* message);
void log_error(const int verbosity, const char* message);
void log_info(const int verbosity, const char* message);
void log_warn(const int verbosity, const char* message);
void log_debug(const int verbosity, const char* message);
void log_verbose(const int verbosity, const char* message);