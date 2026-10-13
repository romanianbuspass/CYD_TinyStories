#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// PLE model format (matches actual on-device model.bin / metadata.json,
// NOT the slvDev upstream PLE1 draft spec — this checkpoint's export tool
// used a different header layout, confirmed by hex dump + metadata.json).
#define TS_MODEL_MAGIC 0x00454C50u /* "PLE\0" */
#define TS_MODEL_HEADER_BYTES 56

typedef struct {
    int32_t version;
    int32_t header_bytes;
    int32_t flags;        /* bit0: TIED_HEAD */
    int32_t input_vocab;  /* embedding table rows (32768) */
    int32_t output_vocab; /* active/output vocab (25353) */
    int32_t d_model;
    int32_t n_layers;
    int32_t n_heads;
    int32_t ffn_hidden;
    int32_t ple_dim;
    int32_t seq_len;
    int32_t group; /* quant group size, 128 */
    float   rope_theta;
} ts_config_t;

typedef struct {
    ts_config_t cfg;
    const uint8_t *base;   /* mmap'd flash base of model partition */
    size_t         size;
    bool           loaded;
    void          *tensors; /* opaque bound-tensor table (ts_bound_model_t*), owned */
} ts_model_t;

// Opaque run state (KV cache, activation scratch) allocated in PSRAM
typedef struct ts_run_state ts_run_state_t;

/* Loads model header + validates magic from a memory-mapped flash region. */
bool ts_model_load(ts_model_t *m, const void *mmap_base, size_t mmap_size);
void ts_model_free(ts_model_t *m);

/* Allocates transformer run state (KV cache + activations) in PSRAM. */
ts_run_state_t *ts_run_state_create(const ts_model_t *m);
void ts_run_state_free(ts_run_state_t *rs);

/* Tokenizer: loads BPE vocab from embedded/flashed tokenizer table. */
bool ts_tokenizer_init(void);
int  ts_tokenizer_encode(const char *text, int *out_ids, int max_ids);
const char *ts_tokenizer_decode(int token_id);

/* Runs one forward pass for `token` at position `pos`, returns logits pointer
 * (vocab_size floats, owned by run state scratch — valid until next call). */
const float *ts_forward(const ts_model_t *m, ts_run_state_t *rs, int token, int pos);

/* Samples next token from logits (temperature/top-p handled internally). */
int ts_sample(const float *logits, int vocab_size, float temperature, float *scratch);

/* Abort flag: set to true to stop an in-flight generation loop early. */
extern volatile bool ts_abort_requested;

/* Opaque accessor: returns a PSRAM scratch buffer of size >= vocab_size
 * floats, owned by run state, for use by ts_sample (avoids alloca on the
 * caller's stack — vocab_size * 4 bytes is far larger than a task stack). */
float *ts_run_state_sample_scratch(ts_run_state_t *rs);

#ifdef __cplusplus
}
#endif
