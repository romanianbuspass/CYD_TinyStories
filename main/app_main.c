#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "driver/gpio.h"
#include "lvgl.h"

#include "app_shared.h"
#include "board_cyd.h"
#include "st77922.h"
#include "st77922_touch.h"

static const char *TAG = "app_main";

/* Cross-task queues declared extern in app_shared.h */
QueueHandle_t inference_prompt_queue = NULL;
QueueHandle_t token_stream_queue = NULL;

/* Actual hardware (confirmed via Freenove_ESP32_S3/gaiacode_cyd reference):
 * ST77922 QSPI panel + working ST77922 TDDI I2C touch (0x55, SCL=39/SDA=38).
 * Touch DOES work on this board — ported from gaiacode_cyd's Arduino HAL. */
#define CYD_PARTIAL_ROWS 40
#define FB_PIXELS (CYD_LCD_H_RES * CYD_PARTIAL_ROWS)

static lv_disp_draw_buf_t s_draw_buf;
static uint8_t *s_buf1, *s_buf2;

static void tddi_reset(void) {
    const gpio_config_t io = {
        .pin_bit_mask = (1ULL << CYD_TOUCH_RST),
        .mode = GPIO_MODE_OUTPUT,
    };
    gpio_config(&io);
    gpio_set_level(CYD_TOUCH_RST, 1);
    vTaskDelay(pdMS_TO_TICKS(10));
    gpio_set_level(CYD_TOUCH_RST, 0);
    vTaskDelay(pdMS_TO_TICKS(10));
    gpio_set_level(CYD_TOUCH_RST, 1);
    vTaskDelay(pdMS_TO_TICKS(200));
}

static void disp_rounder(lv_disp_drv_t *drv, lv_area_t *area) {
    (void)drv;
    area->x1 &= ~(CYD_QSPI_ALIGN - 1);
    area->y1 &= ~(CYD_QSPI_ALIGN - 1);
    area->x2 |= (CYD_QSPI_ALIGN - 1);
    area->y2 |= (CYD_QSPI_ALIGN - 1);
    if (area->x2 >= CYD_LCD_H_RES) area->x2 = CYD_LCD_H_RES - 1;
    if (area->y2 >= CYD_LCD_V_RES) area->y2 = CYD_LCD_V_RES - 1;
}

static void disp_flush(lv_disp_drv_t *drv, const lv_area_t *area, lv_color_t *color_p) {
    int32_t w = area->x2 - area->x1 + 1;
    int32_t h = area->y2 - area->y1 + 1;

    /* Panel assembles 16-bit pixels high-byte-first: byte-swap before
     * transmit (proven in CYD_STW / gaiacode_cyd bring-up). */
    uint16_t *px = (uint16_t *)color_p;
    size_t n = (size_t)w * (size_t)h;
    for (size_t i = 0; i < n; i++) px[i] = (uint16_t)((px[i] << 8) | (px[i] >> 8));

    st77922_write_pixels((uint16_t)area->x1, (uint16_t)area->y1,
                          (uint16_t)w, (uint16_t)h, (const uint16_t *)color_p, (uint32_t)w);
    lv_disp_flush_ready(drv);
}

/* Touch indev read callback (LVGL8 pointer device). Ported protocol from
 * gaiacode_cyd's touch_read(): first valid point (id[0] bit0x01 set),
 * clamped to panel bounds, rotation-0 direct mapping. */
static void touch_read_cb(lv_indev_drv_t *drv, lv_indev_data_t *data) {
    (void)drv;
    st77922_touch_point_t pt;
    if (st77922_touch_get(&pt) && (pt.id[0] & 0x01)) {
        uint16_t x = pt.x[0] > (CYD_LCD_H_RES - 1) ? (CYD_LCD_H_RES - 1) : pt.x[0];
        uint16_t y = pt.y[0] > (CYD_LCD_V_RES - 1) ? (CYD_LCD_V_RES - 1) : pt.y[0];
        data->point.x = x;
        data->point.y = y;
        data->state = LV_INDEV_STATE_PRESSED;
    } else {
        data->state = LV_INDEV_STATE_RELEASED;
    }
}

/* Core 0: LVGL tick + refresh handler task. */
static void lvgl_ticker_task(void *arg) {
    (void)arg;
    const TickType_t period = pdMS_TO_TICKS(5);
    for (;;) {
        lv_tick_inc(5);
        lv_timer_handler();
        vTaskDelay(period);
    }
}

/* Core 0: drains generated token fragments into the visible textarea,
 * decoupling UI updates from the Core-1 inference producer. */
static void token_stream_consumer_task(void *arg) {
    (void)arg;
    char frag[TOKEN_FRAGMENT_MAX];
    for (;;) {
        if (xQueueReceive(token_stream_queue, frag, portMAX_DELAY) == pdTRUE) {
            ui_append_token(frag);
        }
    }
}

void app_main(void) {
    ESP_LOGI(TAG, "CYD TinyStories starting");

    inference_prompt_queue = xQueueCreate(1, sizeof(const char *));
    token_stream_queue = xQueueCreate(64, TOKEN_FRAGMENT_MAX);
    if (!inference_prompt_queue || !token_stream_queue) {
        ESP_LOGE(TAG, "queue creation failed");
        return;
    }

    tddi_reset();
    st77922_init();
    st77922_touch_init();

    lv_init();
    size_t buf_bytes = FB_PIXELS * sizeof(lv_color_t);
    s_buf1 = heap_caps_malloc(buf_bytes, MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
    s_buf2 = heap_caps_malloc(buf_bytes, MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
    if (!s_buf1 || !s_buf2) {
        ESP_LOGE(TAG, "draw buffer alloc failed");
        return;
    }
    lv_disp_draw_buf_init(&s_draw_buf, s_buf1, s_buf2, FB_PIXELS);

    static lv_disp_drv_t disp_drv;
    lv_disp_drv_init(&disp_drv);
    disp_drv.hor_res = CYD_LCD_H_RES;
    disp_drv.ver_res = CYD_LCD_V_RES;
    disp_drv.flush_cb = disp_flush;
    disp_drv.rounder_cb = disp_rounder;
    disp_drv.draw_buf = &s_draw_buf;
    lv_disp_t *disp = lv_disp_drv_register(&disp_drv);

    static lv_indev_drv_t indev_drv;
    lv_indev_drv_init(&indev_drv);
    indev_drv.type = LV_INDEV_TYPE_POINTER;
    indev_drv.read_cb = touch_read_cb;
    lv_indev_drv_register(&indev_drv);

    init_story_ui(lv_disp_get_scr_act(disp));

    /* Core 0: rendering + UI-side consumer. */
    xTaskCreatePinnedToCore(lvgl_ticker_task, "lvgl_tick", 4096, NULL, 4, NULL, 0);
    xTaskCreatePinnedToCore(token_stream_consumer_task, "tok_consumer", 4096, NULL, 4, NULL, 0);

    /* Core 1: model inference. */
    inference_task_start();
}
