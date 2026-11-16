// The NVTX range API used by h2o4gpu, mapped to roctx.
#pragma once

#include <stdint.h>

#if __has_include(<roctracer/roctx.h>)
#include <roctracer/roctx.h>
#else
// No roctx in this ROCm (Windows): ranges are no-ops.
static inline int roctxRangePushA(const char *) { return 0; }
static inline int roctxRangePop(void) { return 0; }
#endif

#define NVTX_VERSION 2
#define NVTX_EVENT_ATTRIB_STRUCT_SIZE ((uint16_t)sizeof(nvtxEventAttributes_t))
#define NVTX_COLOR_ARGB 1
#define NVTX_MESSAGE_TYPE_ASCII 1

// roctx ranges carry only a message; the other attributes are accepted and
// ignored.
typedef struct nvtxEventAttributes_v2 {
  uint16_t version;
  uint16_t size;
  int32_t colorType;
  uint32_t color;
  int32_t messageType;
  union {
    const char *ascii;
  } message;
} nvtxEventAttributes_t;

static inline int nvtxRangePushEx(const nvtxEventAttributes_t *attrib) {
  return roctxRangePushA(attrib->message.ascii);
}

static inline int nvtxRangePushA(const char *message) {
  return roctxRangePushA(message);
}

static inline int nvtxRangePop(void) { return roctxRangePop(); }
