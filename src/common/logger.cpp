/*!
 * Copyright 2017-2018 H2O.ai, Inc.
 * License   Apache License Version 2.0 (see LICENSE for details)
 */
#include "logger.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>

static const char *levels[] = {
    "NOTHING", "FATAL", "ERROR", "INFO", "WARN", "DEBUG", "VERBOSE"
};

bool should_log(const int desired_lvl, const int verbosity) {
  return verbosity > H2O4GPU_LOG_NOTHING && verbosity <= desired_lvl;
}

void log(int desired_level, int level, const char *file, int line, const char *fmt, ...) {
  if (should_log(desired_level, level)) {
    time_t now = time(NULL);
    struct tm *local_time = localtime(&now);

    va_list args;
    char buf[16];
    buf[strftime(buf, sizeof(buf), "%H:%M:%S", local_time)] = '\0';
    fprintf(stderr, "%s %-5s %s:%d: ", buf, levels[level / 100], file, line);
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
  }
}