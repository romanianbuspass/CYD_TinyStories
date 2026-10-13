#pragma once
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "tinystories_engine.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Shared inter-task queues (defined in app_main.c) */
extern QueueHandle_t inference_prompt_queue; /* depth 1, holds const char* */
extern QueueHandle_t token_stream_queue;     /* depth 64, holds char[32] fragments */

#define TOKEN_FRAGMENT_MAX 32

/* flash_model.c */
bool flash_model_mmap(ts_model_t *model_out);

/* inference_task.c */
void inference_task_start(void); /* pins task to Core 1 */

/* ui_story.c */
void init_story_ui(void *lv_parent_obj);
void ui_append_token(const char *token);

#ifdef __cplusplus
}
#endif
