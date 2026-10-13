#pragma once
/* Minimal host stand-ins for the ESP-IDF calls used by ts_model.c so it
 * can be compiled and run on the workstation for sanity verification. */
#include <stdlib.h>
#include <stdio.h>
#define MALLOC_CAP_SPIRAM 0
#define MALLOC_CAP_DMA 0
#define MALLOC_CAP_INTERNAL 0
static inline void *heap_caps_malloc(size_t n, int caps) { (void)caps; return malloc(n); }
static inline void *heap_caps_calloc(size_t n1, size_t n2, int caps) { (void)caps; return calloc(n1, n2); }
static inline void heap_caps_free(void *p) { free(p); }
typedef struct { int dummy; } esp_log_stub_t;
#define ESP_LOGE(tag, fmt, ...) fprintf(stderr, "E(%s) " fmt "\n", tag, ##__VA_ARGS__)
#define ESP_LOGI(tag, fmt, ...) fprintf(stderr, "I(%s) " fmt "\n", tag, ##__VA_ARGS__)
