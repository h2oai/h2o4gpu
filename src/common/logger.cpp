/*!
 * Copyright 2017 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#include "logger.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

bool should_log(const int verbosity, const int desired_lvl) {
  return verbosity > H2O4GPU_LOG_NOTHING && verbosity <= desired_lvl;
}

void log(const char *message) {
  time_t now;
  time(&now);
  fprintf(stderr, "%s: %s\n", ctime(&now), message);
}

void log_fatal(const int verbosity, const char *message) {
  if (should_log(verbosity, H2O4GPU_LOG_FATAL)) {
    log(message);
  }
}

void log_error(const int verbosity, const char *message) {
  if (should_log(verbosity, H2O4GPU_LOG_ERROR)) {
    log(message);
  }
}

void log_info(const int verbosity, const char *message) {
  if (should_log(verbosity, H2O4GPU_LOG_INFO)) {
    log(message);
  }
}

void log_warn(const int verbosity, const char *message) {
  if (should_log(verbosity, H2O4GPU_LOG_WARN)) {
    log(message);
  }
}

void log_debug(const int verbosity, const char *message) {
  if (should_log(verbosity, H2O4GPU_LOG_DEBUG)) {
    log(message);
  }
}

void log_verbose(const int verbosity, const char *message) {
  if (should_log(verbosity, H2O4GPU_LOG_VERBOSE)) {
    log(message);
  }
}