#include "tinystories_engine.h"
#include <string.h>
#include "esp_log.h"

static const char *TAG = "ts_tokenizer";

extern const uint8_t vocab_bin_start[] asm("_binary_vocab_bin_start");
extern const uint8_t vocab_bin_end[]   asm("_binary_vocab_bin_end");

typedef struct { const char *str; uint16_t len; } vocab_entry_t;

static vocab_entry_t *s_vocab = NULL;
static int s_vocab_count = 0;
static char s_decode_scratch[64];

bool ts_tokenizer_init(void) {
    const uint8_t *p = vocab_bin_start;
    uint32_t count;
    memcpy(&count, p, sizeof(count));
    p += sizeof(count);

    s_vocab = (vocab_entry_t *)malloc(count * sizeof(vocab_entry_t));
    if (!s_vocab) {
        ESP_LOGE(TAG, "OOM allocating vocab table (%u entries)", (unsigned)count);
        return false;
    }
    for (uint32_t i = 0; i < count; i++) {
        uint16_t len;
        memcpy(&len, p, sizeof(len));
        p += sizeof(len);
        s_vocab[i].str = (const char *)p;
        s_vocab[i].len = len;
        p += len;
        if (p > vocab_bin_end) {
            ESP_LOGE(TAG, "vocab.bin truncated at entry %u", (unsigned)i);
            return false;
        }
    }
    s_vocab_count = (int)count;
    ESP_LOGI(TAG, "tokenizer loaded: %d entries", s_vocab_count);
    return true;
}

/* NOTE(tensor_layout parity): full BPE merge encoding is not yet ported;
 * this does a greedy longest-prefix-match lookup against the vocab table,
 * sufficient for exact-match seed words but not general subword merging.
 * TODO: port merges table from tokenizer.json for arbitrary user text. */
int ts_tokenizer_encode(const char *text, int *out_ids, int max_ids) {
    if (!s_vocab || !text) return 0;
    int n = 0;
    size_t len = strlen(text);
    size_t i = 0;
    while (i < len && n < max_ids) {
        int best_id = -1, best_len = 0;
        for (int v = 0; v < s_vocab_count; v++) {
            int vl = s_vocab[v].len;
            if (vl > best_len && (i + (size_t)vl) <= len &&
                memcmp(text + i, s_vocab[v].str, vl) == 0) {
                best_len = vl;
                best_id = v;
            }
        }
        if (best_id < 0) { i++; continue; } /* skip unknown byte */
        out_ids[n++] = best_id;
        i += best_len;
    }
    return n;
}

const char *ts_tokenizer_decode(int token_id) {
    if (!s_vocab || token_id < 0 || token_id >= s_vocab_count) return "";
    int len = s_vocab[token_id].len;
    if (len >= (int)sizeof(s_decode_scratch)) len = sizeof(s_decode_scratch) - 1;
    memcpy(s_decode_scratch, s_vocab[token_id].str, len);
    s_decode_scratch[len] = '\0';

    /* GPT-2 style BPE vocab encodes word-boundary space as U+0120 'Ġ'
     * (UTF-8 0xC4 0xA0) and newline as U+010A 'Ċ' (0xC4 0x8A). LVGL's
     * built-in Montserrat font has no glyphs for these -> renders as a
     * "tofu" box on screen. Rewrite them to plain ASCII in place so the
     * decoded fragment only ever contains renderable bytes. */
    char *out = s_decode_scratch;
    int n = 0;
    for (int i = 0; i < len; ) {
        if ((unsigned char)s_decode_scratch[i] == 0xC4 && i + 1 < len) {
            unsigned char c2 = (unsigned char)s_decode_scratch[i + 1];
            if (c2 == 0xA0) { out[n++] = ' ';  i += 2; continue; }  /* Ġ -> space */
            if (c2 == 0x8A) { out[n++] = '\n'; i += 2; continue; }  /* Ċ -> newline */
        }
        out[n++] = s_decode_scratch[i++];
    }
    out[n] = '\0';
    return s_decode_scratch;
}
