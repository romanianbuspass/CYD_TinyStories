#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tinystories_engine.h"

static uint8_t *read_file(const char *path, size_t *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc(n);
    if (fread(buf, 1, n, f) != (size_t)n) { perror("fread"); exit(1); }
    fclose(f);
    *n_out = (size_t)n;
    return buf;
}

int main(int argc, char **argv) {
    const char *bin = argc > 1 ? argv[1] : "../model_src/model.bin";
    size_t n;
    uint8_t *buf = read_file(bin, &n);

    ts_model_t m;
    if (!ts_model_load(&m, buf, n)) {
        fprintf(stderr, "model load failed\n");
        return 1;
    }
    ts_run_state_t *rs = ts_run_state_create(&m);
    if (!rs) { fprintf(stderr, "run state alloc failed\n"); return 1; }

    /* Fixed arbitrary prompt token ids (same style as export.py's golden
     * generator: arbitrary ids, no real tokenizer round-trip needed for a
     * pure numerics sanity pass). */
    int prompt[8] = {1, 500, 1000, 200, 42, 777, 13, 99};
    int plen = 8;
    /* Clamp to this checkpoint's input vocab so we don't read OOB rows. */
    for (int i = 0; i < plen; i++) if (prompt[i] >= m.cfg.input_vocab) prompt[i] = 1;

    const float *logits = NULL;
    for (int pos = 0; pos < plen; pos++) {
        logits = ts_forward(&m, rs, prompt[pos], pos);
        if (!logits) { fprintf(stderr, "forward failed at pos=%d\n", pos); return 1; }
    }

    int V = m.cfg.output_vocab;
    int n_nan = 0, n_inf = 0;
    float minv = logits[0], maxv = logits[0];
    double sum = 0, sum2 = 0;
    int top = 0;
    for (int i = 0; i < V; i++) {
        float v = logits[i];
        if (isnan(v)) n_nan++;
        if (isinf(v)) n_inf++;
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
        sum += v; sum2 += (double)v * v;
        if (v > logits[top]) top = i;
    }
    double mean = sum / V;
    double var = sum2 / V - mean * mean;
    printf("output_vocab=%d\n", V);
    printf("nan=%d inf=%d\n", n_nan, n_inf);
    printf("min=%.4f max=%.4f mean=%.4f std=%.4f\n", minv, maxv, mean, sqrt(var > 0 ? var : 0));
    printf("argmax token id = %d (logit=%.4f)\n", top, logits[top]);

    /* Sanity checks: no NaN/Inf, and the distribution isn't degenerate
     * (all-zero or all-identical would indicate a wiring bug, e.g. our
     * earlier all-zero stub). */
    int degenerate = (maxv - minv) < 1e-6f;
    printf(degenerate ? "FAIL: logits degenerate (near-constant)\n"
                       : "PASS: logits finite and non-degenerate\n");

    ts_run_state_free(rs);
    ts_model_free(&m);
    free(buf);
    return (n_nan || n_inf || degenerate) ? 2 : 0;
}
