#include "app_shared.h"
#include "tinystories_engine.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include <string.h>

static const char *TAG = "inference_task";

static ts_model_t s_model;
static bool s_model_ready = false;

static void inference_task_fn(void *arg) {
    (void)arg;
    ts_run_state_t *rs = NULL;

    if (!flash_model_mmap(&s_model)) {
        ESP_LOGE(TAG, "model mmap/load failed; inference task idling");
        vTaskDelete(NULL);
        return;
    }
    rs = ts_run_state_create(&s_model);
    if (!rs) {
        ESP_LOGE(TAG, "failed to allocate PSRAM run state");
        vTaskDelete(NULL);
        return;
    }
    if (!ts_tokenizer_init()) {
        ESP_LOGE(TAG, "tokenizer init failed");
    }
    s_model_ready = true;
    ESP_LOGI(TAG, "inference task ready on core %d", xPortGetCoreID());

    const char *prompt = NULL;
    for (;;) {
        /* Block until a seed prompt is dispatched from the UI task. */
        if (xQueueReceive(inference_prompt_queue, &prompt, portMAX_DELAY) != pdTRUE) {
            continue;
        }
        ts_abort_requested = false;

        int ids[64];
        int n_ids = ts_tokenizer_encode(prompt, ids, 64);
        int pos = 0;
        int token = (n_ids > 0) ? ids[0] : 1;

        /* Prime the KV cache with the prompt tokens. */
        for (int i = 0; i < n_ids && !ts_abort_requested; i++) {
            ts_forward(&s_model, rs, ids[i], pos);
            pos++;
        }

        /* Autoregressive generation loop until seq_len, EOS, or abort. */
        int64_t gen_start_us = esp_timer_get_time();
        int64_t sum_fwd_us = 0;
        int n_tok_timed = 0;
        while (pos < s_model.cfg.seq_len && !ts_abort_requested) {
            int64_t t0 = esp_timer_get_time();
            const float *logits = ts_forward(&s_model, rs, token, pos);
            int64_t t1 = esp_timer_get_time();
            if (!logits) break;
            sum_fwd_us += (t1 - t0);
            n_tok_timed++;
            ESP_LOGI(TAG, "token pos=%d forward_us=%lld", pos, (long long)(t1 - t0));
            token = ts_sample(logits, s_model.cfg.output_vocab, 0.8f, ts_run_state_sample_scratch(rs));

            const char *piece = ts_tokenizer_decode(token);
            if (piece && piece[0] != '\0') {
                char frag[TOKEN_FRAGMENT_MAX];
                strncpy(frag, piece, TOKEN_FRAGMENT_MAX - 1);
                frag[TOKEN_FRAGMENT_MAX - 1] = '\0';
                /* Non-blocking-ish send with a short timeout so an abort
                 * request (new seed tapped) is still responsive. */
                xQueueSend(token_stream_queue, frag, pdMS_TO_TICKS(50));
            }

            /* token id 0 == <|endoftext|> in this tokenizer */
            if (token == 0) break;

            pos++;

            /* Yield periodically so the abort flag / new prompt can preempt. */
            if ((pos & 0x7) == 0) {
                /* Check for a newly queued prompt without blocking. */
                const char *maybe_new = NULL;
                if (xQueuePeek(inference_prompt_queue, &maybe_new, 0) == pdTRUE &&
                    maybe_new != prompt) {
                    ts_abort_requested = true;
                }
            }
        }
        int64_t gen_total_us = esp_timer_get_time() - gen_start_us;
        if (n_tok_timed > 0) {
            ESP_LOGI(TAG, "generation finished/aborted at pos=%d: %d tokens, "
                     "avg_forward_us=%lld total_us=%lld (%.2f tok/s)",
                     pos, n_tok_timed, (long long)(sum_fwd_us / n_tok_timed),
                     (long long)gen_total_us,
                     n_tok_timed / (gen_total_us / 1e6));
        } else {
            ESP_LOGI(TAG, "generation finished/aborted at pos=%d", pos);
        }
    }
}

void inference_task_start(void) {
    /* Pin to Core 1; large stack for activation buffers even though the
     * bulk of tensors live in PSRAM via heap_caps allocations. */
    xTaskCreatePinnedToCore(inference_task_fn, "ts_infer", 8192, NULL,
                             5 /* priority */, NULL, 1 /* Core 1 */);
}
