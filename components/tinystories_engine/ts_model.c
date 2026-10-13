#include "tinystories_engine.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include "esp_heap_caps.h"
#include "esp_log.h"

static const char *TAG = "ts_engine";

volatile bool ts_abort_requested = false;

#define RMS_EPS 1e-6f

/* ---- Quantized tensor view: ragged int4 codes (2/byte) + per-group fp16
 * scales, bound in place from mmap'd flash. Layout/semantics match the
 * upstream slvDev/esp32-ai firmware/common/llm.h exactly (port target for
 * bit-compatible golden-logit verification). */
typedef struct {
    const uint8_t  *codes;   /* rows*row_bytes, nibble = value+8 */
    const uint16_t *scales;  /* rows*n_groups fp16 */
    int rows, cols, group, n_groups, row_bytes;
} qt_t;

/* Bound model: one entry per tensor in export.py's fixed serialization
 * order. n_layers is read from the header (<=8 supported here; this
 * checkpoint has 6). */
#define TS_MAX_LAYERS 8
typedef struct {
    qt_t tok_emb;              /* [Vin, D] tied input embed + output head */
    qt_t ple_model_proj;       /* [L*P, D] */
    const float *ple_proj_norm;/* [P] */
    qt_t ple_table;            /* [Vin, L*P] */
    const float *attn_norm[TS_MAX_LAYERS]; /* [D] */
    qt_t qkv[TS_MAX_LAYERS];               /* [3D, D] */
    qt_t attn_proj[TS_MAX_LAYERS];         /* [D, D] */
    const float *ffn_norm[TS_MAX_LAYERS];  /* [D] */
    qt_t gate[TS_MAX_LAYERS], up[TS_MAX_LAYERS], down[TS_MAX_LAYERS];
    qt_t ple_gate[TS_MAX_LAYERS];   /* [P, D] */
    qt_t ple_proj[TS_MAX_LAYERS];   /* [D, P] */
    const float *ple_norm[TS_MAX_LAYERS]; /* [D] */
    const float *out_norm;    /* [D] */
} ts_bound_model_t;

struct ts_run_state {
    float *x;        /* d_model activation buffer */
    float *h;        /* scratch, d_model */
    float *qkv;      /* 3*d_model */
    float *att;      /* d_model (H*Dh) attention output */
    float *g1, *g2;  /* ffn_hidden scratch (also reused for P-sized ple gate) */
    float *ple;      /* L*P per-layer table gate values */
    float *tmpP;     /* L*P scratch (proj + rope cos/sin reuse) */
    float *trow;     /* L*P scratch (dequant row) */
    float *scores;   /* seq_len, reused per attention head */
    float *logits;   /* output_vocab */
    float *sample_scratch; /* output_vocab, PSRAM — used by ts_sample */
    float *k_cache;  /* n_layers * seq_len * d_model */
    float *v_cache;
    int    pos_max;
    int    vocab_size;
};

/* fp16 -> fp32 (software, no F16C assumed on Xtensa) */
static inline float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign << 31; }
        else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            f = (sign << 31) | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &f, sizeof(out));
    return out;
}

/* ---- tensor binding: mirrors upstream bind_q/bind_f cursor advance ---- */
static const uint8_t *bind_q(const uint8_t *p, qt_t *t, int rows, int cols) {
    int32_t group;
    memcpy(&group, p, 4);
    p += 4;
    t->rows = rows;
    t->cols = cols;
    t->group = group;
    t->n_groups = (cols + group - 1) / group;
    t->row_bytes = (cols + 1) / 2;
    t->codes = p;
    p += (size_t)rows * t->row_bytes;
    t->scales = (const uint16_t *)p;
    p += (size_t)rows * t->n_groups * 2;
    return p;
}
static const uint8_t *bind_f(const uint8_t *p, const float **t, int n) {
    *t = (const float *)p;
    return p + (size_t)n * sizeof(float);
}

/* Dequantize row r of a quant tensor into out[cols]. */
static void deq_row(const qt_t *t, int r, float *out) {
    const uint8_t *row = t->codes + (size_t)r * t->row_bytes;
    const uint16_t *sc = t->scales + (size_t)r * t->n_groups;
    for (int gi = 0; gi < t->n_groups; gi++) {
        int begin = gi * t->group;
        int end = begin + t->group;
        if (end > t->cols) end = t->cols;
        float scale = half_to_float(sc[gi]);
        for (int j = begin; j < end; j++) {
            uint8_t byte = row[j >> 1];
            int code = (j & 1) ? (byte >> 4) : (byte & 0x0F);
            out[j] = (float)(code - 8) * scale;
        }
    }
}

/* y[rows] = W * x[cols], W a quant tensor. Unrolled 2-per-iteration over
 * byte pairs (low nibble, high nibble) to remove the per-element branch
 * that dominated the naive version; this loop is the hot path (~90% of
 * per-token compute is in the output head + qkv/ffn matvecs). */
static void matvec_q(const qt_t *t, const float *x, float *y) {
    for (int r = 0; r < t->rows; r++) {
        const uint8_t *row = t->codes + (size_t)r * t->row_bytes;
        const uint16_t *sc = t->scales + (size_t)r * t->n_groups;
        float acc = 0.0f;
        for (int gi = 0; gi < t->n_groups; gi++) {
            int begin = gi * t->group;
            int end = begin + t->group;
            if (end > t->cols) end = t->cols;
            float scale = half_to_float(sc[gi]);
            float gacc = 0.0f;
            int j = begin;
            /* Fast path: begin is always even (group sizes are powers of
             * two / byte-aligned in this export format), so pairs of
             * codes share one byte load with no branch. */
            int pair_end = end & ~1;
            const uint8_t *bp = row + (begin >> 1);
            for (; j < pair_end; j += 2, bp++) {
                uint8_t byte = *bp;
                float c0 = (float)((int)(byte & 0x0F) - 8);
                float c1 = (float)((int)(byte >> 4) - 8);
                gacc += c0 * x[j] + c1 * x[j + 1];
            }
            for (; j < end; j++) {
                uint8_t byte = row[j >> 1];
                int code = (j & 1) ? (byte >> 4) : (byte & 0x0F);
                gacc += (float)(code - 8) * x[j];
            }
            acc += gacc * scale;
        }
        y[r] = acc;
    }
}

static void rmsnorm(const float *x, const float *w, int n, float *out) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + RMS_EPS);
    for (int i = 0; i < n; i++) out[i] = w[i] * x[i] * inv;
}
static inline float gelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678f)); }
static inline float silu(float x) { return x / (1.0f + expf(-x)); }

static void softmax(float *x, int n) {
    float maxv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > maxv) maxv = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - maxv); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

bool ts_model_load(ts_model_t *m, const void *mmap_base, size_t mmap_size) {
    if (!mmap_base || mmap_size < TS_MODEL_HEADER_BYTES) {
        ESP_LOGE(TAG, "model region too small");
        return false;
    }
    const uint8_t *p = (const uint8_t *)mmap_base;
    uint32_t magic;
    memcpy(&magic, p, sizeof(magic));
    if (magic != TS_MODEL_MAGIC) {
        ESP_LOGE(TAG, "bad magic 0x%08x (expected 0x%08x)", (unsigned)magic, TS_MODEL_MAGIC);
        return false;
    }
    p += sizeof(magic);

    int32_t vals[12];
    memcpy(vals, p, sizeof(vals));
    p += sizeof(vals);
    float rope_theta;
    memcpy(&rope_theta, p, sizeof(rope_theta));
    p += sizeof(rope_theta);

    m->cfg.version      = vals[0];
    m->cfg.header_bytes = vals[1];
    m->cfg.flags        = vals[2];
    m->cfg.input_vocab  = vals[3];
    m->cfg.output_vocab = vals[4];
    m->cfg.d_model      = vals[5];
    m->cfg.n_layers     = vals[6];
    m->cfg.n_heads      = vals[7];
    m->cfg.ffn_hidden   = vals[8];
    m->cfg.ple_dim      = vals[9];
    m->cfg.seq_len      = vals[10];
    m->cfg.group        = vals[11];
    m->cfg.rope_theta   = rope_theta;

    m->base  = mmap_base;
    m->size  = mmap_size;
    m->loaded = true;
    m->tensors = NULL;

    if (m->cfg.n_layers > TS_MAX_LAYERS) {
        ESP_LOGE(TAG, "n_layers=%d exceeds TS_MAX_LAYERS=%d", (int)m->cfg.n_layers, TS_MAX_LAYERS);
        m->loaded = false;
        return false;
    }

    /* Bind every tensor in export.py's fixed order (input_vocab rows for
     * the tied embed/head and the PLE table — NOT output_vocab; confirmed
     * against metadata.json's parameter counts: tok_emb=3,145,728 =
     * 32768*96, ple_table=25,165,824 = 32768*6*128). */
    ts_bound_model_t *bm = heap_caps_calloc(1, sizeof(ts_bound_model_t), MALLOC_CAP_SPIRAM);
    if (!bm) {
        ESP_LOGE(TAG, "OOM allocating tensor binding table");
        m->loaded = false;
        return false;
    }
    int D = m->cfg.d_model, L = m->cfg.n_layers, P = m->cfg.ple_dim, F = m->cfg.ffn_hidden;
    int Vin = m->cfg.input_vocab;

    p = bind_q(p, &bm->tok_emb, Vin, D);
    p = bind_q(p, &bm->ple_model_proj, L * P, D);
    p = bind_f(p, &bm->ple_proj_norm, P);
    p = bind_q(p, &bm->ple_table, Vin, L * P);
    for (int i = 0; i < L; i++) {
        p = bind_f(p, &bm->attn_norm[i], D);
        p = bind_q(p, &bm->qkv[i], 3 * D, D);
        p = bind_q(p, &bm->attn_proj[i], D, D);
        p = bind_f(p, &bm->ffn_norm[i], D);
        p = bind_q(p, &bm->gate[i], F, D);
        p = bind_q(p, &bm->up[i], F, D);
        p = bind_q(p, &bm->down[i], D, F);
        p = bind_q(p, &bm->ple_gate[i], P, D);
        p = bind_q(p, &bm->ple_proj[i], D, P);
        p = bind_f(p, &bm->ple_norm[i], D);
    }
    p = bind_f(p, &bm->out_norm, D);

    size_t consumed = (size_t)(p - (const uint8_t *)mmap_base);
    if (consumed > mmap_size) {
        ESP_LOGE(TAG, "tensor table overruns mmap region (%u > %u)",
                 (unsigned)consumed, (unsigned)mmap_size);
        heap_caps_free(bm);
        m->loaded = false;
        return false;
    }
    m->tensors = bm;

    ESP_LOGI(TAG, "loaded PLE model v%d: Vin=%d Vout=%d D=%d L=%d H=%d F=%d P=%d seq=%d group=%d bytes=%u/%u",
             (int)m->cfg.version, (int)m->cfg.input_vocab, (int)m->cfg.output_vocab,
             (int)m->cfg.d_model, (int)m->cfg.n_layers, (int)m->cfg.n_heads,
             (int)m->cfg.ffn_hidden, (int)m->cfg.ple_dim, (int)m->cfg.seq_len, (int)m->cfg.group,
             (unsigned)consumed, (unsigned)mmap_size);
    return true;
}

void ts_model_free(ts_model_t *m) {
    if (!m) return;
    heap_caps_free(m->tensors);
    m->tensors = NULL;
    m->loaded = false;
}

ts_run_state_t *ts_run_state_create(const ts_model_t *m) {
    ts_run_state_t *rs = heap_caps_calloc(1, sizeof(ts_run_state_t), MALLOC_CAP_SPIRAM);
    if (!rs) return NULL;
    int d = m->cfg.d_model;
    int L = m->cfg.n_layers;
    int P = m->cfg.ple_dim;
    int F = m->cfg.ffn_hidden;
    int S = m->cfg.seq_len;
    int V = m->cfg.output_vocab;

    rs->x     = heap_caps_malloc(d * sizeof(float), MALLOC_CAP_SPIRAM);
    rs->h     = heap_caps_malloc(d * sizeof(float), MALLOC_CAP_SPIRAM);
    rs->qkv   = heap_caps_malloc(3 * d * sizeof(float), MALLOC_CAP_SPIRAM);
    rs->att   = heap_caps_malloc(d * sizeof(float), MALLOC_CAP_SPIRAM);
    int gmax  = (F > P) ? F : P;
    rs->g1    = heap_caps_malloc(gmax * sizeof(float), MALLOC_CAP_SPIRAM);
    rs->g2    = heap_caps_malloc(gmax * sizeof(float), MALLOC_CAP_SPIRAM);
    rs->ple   = heap_caps_malloc((size_t)L * P * sizeof(float), MALLOC_CAP_SPIRAM);
    rs->tmpP  = heap_caps_malloc((size_t)L * P * sizeof(float), MALLOC_CAP_SPIRAM);
    rs->trow  = heap_caps_malloc((size_t)L * P * sizeof(float), MALLOC_CAP_SPIRAM);
    rs->scores = heap_caps_malloc((size_t)S * sizeof(float), MALLOC_CAP_SPIRAM);
    rs->logits = heap_caps_malloc((size_t)V * sizeof(float), MALLOC_CAP_SPIRAM);
    rs->sample_scratch = heap_caps_malloc((size_t)V * sizeof(float), MALLOC_CAP_SPIRAM);
    rs->k_cache = heap_caps_calloc((size_t)L * S * d, sizeof(float), MALLOC_CAP_SPIRAM);
    rs->v_cache = heap_caps_calloc((size_t)L * S * d, sizeof(float), MALLOC_CAP_SPIRAM);
    rs->pos_max = S;
    rs->vocab_size = V;

    if (!rs->x || !rs->h || !rs->qkv || !rs->att || !rs->g1 || !rs->g2 || !rs->ple ||
        !rs->tmpP || !rs->trow || !rs->scores || !rs->logits || !rs->sample_scratch ||
        !rs->k_cache || !rs->v_cache) {
        ts_run_state_free(rs);
        return NULL;
    }
    return rs;
}

void ts_run_state_free(ts_run_state_t *rs) {
    if (!rs) return;
    heap_caps_free(rs->x);
    heap_caps_free(rs->h);
    heap_caps_free(rs->qkv);
    heap_caps_free(rs->att);
    heap_caps_free(rs->g1);
    heap_caps_free(rs->g2);
    heap_caps_free(rs->ple);
    heap_caps_free(rs->tmpP);
    heap_caps_free(rs->trow);
    heap_caps_free(rs->scores);
    heap_caps_free(rs->logits);
    heap_caps_free(rs->sample_scratch);
    heap_caps_free(rs->k_cache);
    heap_caps_free(rs->v_cache);
    heap_caps_free(rs);
}

/* Full forward pass, ported op-for-op from slvDev/esp32-ai's
 * firmware/common/llm.h: embed -> per-layer input gate (PLE) -> attention
 * (split-half RoPE) -> SwiGLU FFN -> PLE gate -> final RMSNorm -> tied
 * output head. Not yet verified against golden.txt bit-for-bit, but the
 * structure, tensor order, and math match the reference exactly. */
const float *ts_forward(const ts_model_t *m, ts_run_state_t *rs, int token, int pos) {
    if (!m->loaded || !m->tensors) return NULL;
    const ts_bound_model_t *bm = (const ts_bound_model_t *)m->tensors;

    int D = m->cfg.d_model, L = m->cfg.n_layers, P = m->cfg.ple_dim, F = m->cfg.ffn_hidden;
    int H = m->cfg.n_heads, Dh = D / H, S = m->cfg.seq_len;
    if (Dh < 2 || (Dh & 1)) {
        /* RoPE needs an even head dim; bail safely rather than corrupt memory. */
        ESP_LOGE(TAG, "unsupported head dim Dh=%d (D=%d H=%d)", Dh, D, H);
        return NULL;
    }
    if (token < 0 || token >= m->cfg.input_vocab) token = 0;
    if (pos >= S) pos = S - 1;

    deq_row(&bm->tok_emb, token, rs->x); /* embedding */

    /* per-layer input: (RMSNorm(proj(x)/sqrt(D)) + table[tok]*sqrt(P)) / sqrt(2) */
    matvec_q(&bm->ple_model_proj, rs->x, rs->tmpP); /* [L*P] */
    float dscale = 1.0f / sqrtf((float)D);
    for (int i = 0; i < L * P; i++) rs->tmpP[i] *= dscale;
    for (int l = 0; l < L; l++)
        rmsnorm(rs->tmpP + l * P, bm->ple_proj_norm, P, rs->tmpP + l * P);
    deq_row(&bm->ple_table, token, rs->trow); /* [L*P] */
    float sp = sqrtf((float)P), inv2 = 0.70710678f;
    for (int i = 0; i < L * P; i++)
        rs->ple[i] = (rs->tmpP[i] + rs->trow[i] * sp) * inv2;

    /* RoPE cos/sin for this position (same across layers/heads); reuse
     * trow (dead after building rs->ple) instead of a fresh buffer. */
    float *rope_c = rs->trow, *rope_s = rs->trow + Dh / 2;
    for (int i = 0; i < Dh / 2; i++) {
        float freq = powf(m->cfg.rope_theta, -2.0f * i / Dh);
        rope_c[i] = cosf(pos * freq);
        rope_s[i] = sinf(pos * freq);
    }

    for (int l = 0; l < L; l++) {
        /* ---- attention ---- */
        rmsnorm(rs->x, bm->attn_norm[l], D, rs->h);
        matvec_q(&bm->qkv[l], rs->h, rs->qkv); /* [3D] */
        float *q = rs->qkv, *k = rs->qkv + D, *v = rs->qkv + 2 * D;
        for (int hh = 0; hh < H; hh++) {
            float *qh = q + hh * Dh, *kh = k + hh * Dh;
            for (int i = 0; i < Dh / 2; i++) {
                float c = rope_c[i], sn = rope_s[i];
                float q1 = qh[i], q2 = qh[i + Dh / 2];
                qh[i] = q1 * c - q2 * sn; qh[i + Dh / 2] = q2 * c + q1 * sn;
                float k1 = kh[i], k2 = kh[i + Dh / 2];
                kh[i] = k1 * c - k2 * sn; kh[i + Dh / 2] = k2 * c + k1 * sn;
            }
        }
        float *kc = rs->k_cache + (size_t)l * S * D, *vc = rs->v_cache + (size_t)l * S * D;
        memcpy(kc + (size_t)pos * D, k, D * sizeof(float));
        memcpy(vc + (size_t)pos * D, v, D * sizeof(float));

        float scale = 1.0f / sqrtf((float)Dh);
        for (int hh = 0; hh < H; hh++) {
            float *qh = q + hh * Dh;
            float *ao = rs->att + hh * Dh;
            for (int i = 0; i < Dh; i++) ao[i] = 0.0f;
            float maxs = -1e30f;
            for (int t = 0; t <= pos; t++) {
                float *kt = kc + (size_t)t * D + hh * Dh, dot = 0.0f;
                for (int i = 0; i < Dh; i++) dot += qh[i] * kt[i];
                dot *= scale;
                rs->scores[t] = dot;
                if (dot > maxs) maxs = dot;
            }
            float denom = 0.0f;
            for (int t = 0; t <= pos; t++) {
                float w = expf(rs->scores[t] - maxs); denom += w;
                float *vt = vc + (size_t)t * D + hh * Dh;
                for (int i = 0; i < Dh; i++) ao[i] += w * vt[i];
            }
            for (int i = 0; i < Dh; i++) ao[i] /= denom;
        }
        matvec_q(&bm->attn_proj[l], rs->att, rs->h);
        for (int i = 0; i < D; i++) rs->x[i] += rs->h[i];

        /* ---- SwiGLU FFN ---- */
        rmsnorm(rs->x, bm->ffn_norm[l], D, rs->h);
        matvec_q(&bm->gate[l], rs->h, rs->g1);
        matvec_q(&bm->up[l], rs->h, rs->g2);
        for (int i = 0; i < F; i++) rs->g1[i] = silu(rs->g1[i]) * rs->g2[i];
        matvec_q(&bm->down[l], rs->g1, rs->h);
        for (int i = 0; i < D; i++) rs->x[i] += rs->h[i];

        /* ---- PLE gate: x += RMSNorm(ple_proj(gelu(ple_gate(x)) * ple_l)) */
        matvec_q(&bm->ple_gate[l], rs->x, rs->g2); /* [P] */
        for (int i = 0; i < P; i++) rs->g2[i] = gelu(rs->g2[i]) * rs->ple[l * P + i];
        matvec_q(&bm->ple_proj[l], rs->g2, rs->h); /* [D] */
        rmsnorm(rs->h, bm->ple_norm[l], D, rs->h);
        for (int i = 0; i < D; i++) rs->x[i] += rs->h[i];
    }

    rmsnorm(rs->x, bm->out_norm, D, rs->x);
    /* Tied output head: only the first output_vocab rows of tok_emb are
     * ever sampled (checkpoint's active/output vocab is a prefix of the
     * full 32768-row input embedding table). Reuse the optimized
     * matvec_q inner loop by giving it a row-count-limited view. */
    qt_t head_view = bm->tok_emb;
    head_view.rows = m->cfg.output_vocab;
    matvec_q(&head_view, rs->x, rs->logits);
    return rs->logits;
}

int ts_sample(const float *logits, int vocab_size, float temperature, float *scratch) {
    if (temperature <= 0.0f) {
        int best = 0; float bv = logits[0];
        for (int i = 1; i < vocab_size; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
        return best;
    }
    for (int i = 0; i < vocab_size; i++) scratch[i] = logits[i] / temperature;
    softmax(scratch, vocab_size);
    float r = (float)rand() / (float)RAND_MAX;
    float cdf = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cdf += scratch[i];
        if (r < cdf) return i;
    }
    return vocab_size - 1;
}

float *ts_run_state_sample_scratch(ts_run_state_t *rs) {
    return rs ? rs->sample_scratch : NULL;
}
