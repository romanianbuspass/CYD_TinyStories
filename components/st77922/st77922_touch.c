/*
 * ST77922 TDDI I2C touch — ESP-IDF port of Freenove_ESP32_S3/gaiacode_cyd's
 * ST77922_Touch.{h,cpp}. Register map / read protocol kept verbatim; Wire
 * calls replaced with driver/i2c native i2c_master (I2C_NUM_0, 100 kHz).
 */
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c.h"
#include "driver/gpio.h"
#include "esp_log.h"

#include "st77922_touch.h"
#include "board_cyd.h"

static const char *TAG = "st77922_touch";

#define TOUCH_I2C_PORT   I2C_NUM_0
#define TOUCH_I2C_SPEED  100000

#define FW_VER       0x0000
#define STATUS       0x0001
#define MAX_TOUCHES  0x0009
#define TOUCH_INFO   0x0010
#define TOUCH_POINT0 0x0014

static uint8_t s_max_points;

static void touch_read_reg(uint16_t reg, uint8_t *rbuf, size_t rlen) {
    uint8_t addr_bytes[2] = { (uint8_t)((reg >> 8) & 0xFF), (uint8_t)(reg & 0xFF) };
    i2c_master_write_read_device(TOUCH_I2C_PORT, CYD_TOUCH_ADDR,
                                  addr_bytes, sizeof(addr_bytes),
                                  rbuf, rlen, pdMS_TO_TICKS(50));
}

static void touch_reset(void) {
    gpio_set_level(CYD_TOUCH_RST, 0);
    vTaskDelay(pdMS_TO_TICKS(100));
    gpio_set_level(CYD_TOUCH_RST, 1);
    vTaskDelay(pdMS_TO_TICKS(100));
}

void st77922_touch_init(void) {
    const gpio_config_t rst_io = {
        .pin_bit_mask = (1ULL << CYD_TOUCH_RST),
        .mode = GPIO_MODE_OUTPUT,
    };
    gpio_config(&rst_io);
    const gpio_config_t int_io = {
        .pin_bit_mask = (1ULL << CYD_TOUCH_INT),
        .mode = GPIO_MODE_INPUT,
    };
    gpio_config(&int_io);

    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = CYD_TOUCH_SDA,
        .scl_io_num = CYD_TOUCH_SCL,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = TOUCH_I2C_SPEED,
    };
    esp_err_t err = i2c_param_config(TOUCH_I2C_PORT, &conf);
    if (err == ESP_OK) err = i2c_driver_install(TOUCH_I2C_PORT, conf.mode, 0, 0, 0);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "i2c init failed: %d", err);
        return;
    }

    touch_reset();

    uint8_t data;
    int guard = 0;
    do {
        touch_read_reg(STATUS, &data, 1);
        if (++guard > 200) { ESP_LOGW(TAG, "status poll timeout"); break; }
        vTaskDelay(pdMS_TO_TICKS(5));
    } while (data & 0x0F);

    touch_read_reg(MAX_TOUCHES, &data, 1);
    s_max_points = data;
    if (s_max_points == 0 || s_max_points > ST77922_TOUCH_MAX_POINTS) {
        ESP_LOGW(TAG, "unexpected max_points=%u, clamping to 1", s_max_points);
        s_max_points = 1;
    }
    ESP_LOGI(TAG, "touch up: max_points=%u addr=0x%02x", s_max_points, CYD_TOUCH_ADDR);
}

uint8_t st77922_touch_max_points(void) { return s_max_points; }

bool st77922_touch_get(st77922_touch_point_t *out) {
    uint8_t update = 0;
    touch_read_reg(TOUCH_INFO, &update, 1);
    if (!(update & 0x08)) return false;

    uint8_t data[7 * ST77922_TOUCH_MAX_POINTS] = {0};
    touch_read_reg(TOUCH_POINT0, data, (size_t)7 * s_max_points);

    memset(out, 0, sizeof(*out));
    for (uint8_t i = 0; i < s_max_points; i++) {
        if (data[i * 7] & 0x80) {
            out->id[i] = (data[i * 7] & 0x80) >> 7;
            /* rotation 0 mapping only — matches board's fixed MADCTL 0x00 */
            out->x[i] = (uint16_t)(((data[i * 7] & 0x3F) << 8) | data[i * 7 + 1]);
            out->y[i] = (uint16_t)(((data[i * 7 + 2] & 0x3F) << 8) | data[i * 7 + 3]);
        }
    }
    return true;
}
