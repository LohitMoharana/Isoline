/* Streaming wrapper: feed one 360 Hz slot of comparator event counts at a time; get classified beats out.
 * Produces exactly the beats of iso_qrs_detect() and exactly the features of iso_beat_features() (verified by
 * firmware/test/verify_stream.py). Latency: a beat is classified 160 slots (444 ms) after its R peak. */
#ifndef ISOLINE_STREAM_H
#define ISOLINE_STREAM_H
#include "isoline_core.h"

#define ISO_RB 512                  /* ring buffer (slots), power of two; >= 257 + detector look-back */
#define ISO_PENDING 16

typedef struct {
    int64_t r;                      /* beat position (slot index since start) */
    int32_t feat[ISO_NFEAT];        /* the 216 model inputs */
    float af[ISO_NAF];
    int af_valid;
    int have_feat;                  /* 0 when the beat window started before the first slot */
} iso_beat;

typedef struct {
    int64_t T;                      /* slots received */
    int16_t up[ISO_RB], dn[ISO_RB], acc[ISO_RB];
    int16_t E[ISO_RB], S[ISO_RB];    /* event counts in the 33 ms / 11 ms windows */
    int32_t q[ISO_RB];               /* up/down staircase */
    uint32_t cu[ISO_RB], cd[ISO_RB]; /* prefix sums (mod 2^32; window differences are exact) */
    int64_t run_E, run_S, run_q, run_cu, run_cd;
    iso_qrs_state qs; int64_t t_det; int n_det;
    int64_t pending[ISO_PENDING]; int n_pending;
    int64_t last_r; int have_last;
    iso_feat_state fs;              /* beat features + leaky integrator */
    iso_feat_state as;              /* AF RR history */
} iso_stream;

void iso_stream_init(iso_stream *s);
/* Push one slot. Returns the number of beats written to out (0 or 1 in practice; at most max_out). */
int iso_stream_push(iso_stream *s, int up, int dn, iso_beat *out, int max_out);

#endif
