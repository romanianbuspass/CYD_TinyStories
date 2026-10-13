#include "lvgl.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "app_shared.h"

static lv_obj_t *story_text_area;
static lv_obj_t *seed_matrix;

static const char *seed_map[] = {
    "Once Upon", "Dog Spot", "Dark Forest", "\n",
    "Shiny Ball", "Tiny Robot", "Cat Wish", ""
};

static const char *full_prompts[] = {
    "Once upon a time, ",
    "Tim and his dog Spot found a ",
    "In a dark, quiet forest, ",
    "Lily had a shiny red ball, but ",
    "A tiny robot woke up and ",
    "A little cat wanted to ",
};
#define NUM_SEEDS (sizeof(full_prompts) / sizeof(full_prompts[0]))

static void seed_btn_event_cb(lv_event_t *e) {
    lv_event_code_t code = lv_event_get_code(e);
    lv_obj_t *obj = lv_event_get_target(e);

    if (code == LV_EVENT_VALUE_CHANGED) {
        uint32_t id = lv_btnmatrix_get_selected_btn(obj);
        if (id < NUM_SEEDS) {
            const char *prompt = full_prompts[id];
            /* Clear current story and dispatch prompt. xQueueOverwrite
             * both refreshes a stalled queue-of-1 and signals the
             * inference task's peek-check to set ts_abort_requested. */
            lv_textarea_set_text(story_text_area, "");
            if (inference_prompt_queue != NULL) {
                xQueueOverwrite(inference_prompt_queue, &prompt);
            }
        }
    }
}

void init_story_ui(void *lv_parent_obj) {
    lv_obj_t *parent = (lv_obj_t *)lv_parent_obj;

    /* 1. Text display region */
    story_text_area = lv_textarea_create(parent);
    lv_obj_set_size(story_text_area, LV_PCT(100), LV_PCT(75));
    lv_obj_align(story_text_area, LV_ALIGN_TOP_MID, 0, 0);
    lv_textarea_set_text(story_text_area, "Tap a seed below to begin a story...\n\n");
    lv_textarea_set_cursor_click_pos(story_text_area, false);
    lv_obj_clear_flag(story_text_area, LV_OBJ_FLAG_CLICK_FOCUSABLE);

    /* 2. Tactile seed buttons — touch-driven (ST77922 TDDI I2C touch,
     * confirmed working on this board). */
    seed_matrix = lv_btnmatrix_create(parent);
    lv_btnmatrix_set_map(seed_matrix, seed_map);
    lv_obj_set_size(seed_matrix, LV_PCT(100), LV_PCT(25));
    lv_obj_align(seed_matrix, LV_ALIGN_BOTTOM_MID, 0, 0);
    lv_obj_add_event_cb(seed_matrix, seed_btn_event_cb, LV_EVENT_VALUE_CHANGED, NULL);
}

void ui_append_token(const char *token) {
    if (story_text_area != NULL && token != NULL) {
        lv_textarea_add_text(story_text_area, token);
    }
}
