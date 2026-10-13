#pragma once
/*
 * ST77922 TDDI I2C touch — ESP-IDF port of Freenove_ESP32_S3/gaiacode_cyd's
 * ST77922_Touch.{h,cpp} (Arduino Wire -> driver/i2c native i2c_master).
 * Confirmed working on this exact CYD board (touch DOES work; earlier
 * CYD_STW reference board apparently didn't have it wired/working).
 */
#include <stdint.h>
#include <stdbool.h>

#define ST77922_TOUCH_MAX_POINTS 10

typedef struct {
    uint8_t  id[ST77922_TOUCH_MAX_POINTS];
    uint16_t x[ST77922_TOUCH_MAX_POINTS];
    uint16_t y[ST77922_TOUCH_MAX_POINTS];
} st77922_touch_point_t;

/* I2C init (bus + TDDI reset + max_points query). Call after st77922_init(). */
void st77922_touch_init(void);

/* Poll the touch controller. On true, `out` holds the latest point set
 * (only entries with (id[i] & 0x80) bit set — i.e. valid — should be used,
 * mirroring the Arduino driver's data[i*7] & 0x80 check). */
bool st77922_touch_get(st77922_touch_point_t *out);

uint8_t st77922_touch_max_points(void);
