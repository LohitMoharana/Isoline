/* Isoline core: platform-independent, integer feature extraction + beat detection (mirrors isoline/features.py
 * and isoline/qrs.py bit-exactly; verified by firmware/test/verify_core.py). No dynamic allocation.
 *
 * Feature vector (ISO_NFEAT = 216 int32 values, the layout the exported tree model expects):
 *   [0..95]    gain-normalised spike histograms h3 (3 channels: up, down, leaky-integrator; 32 bins each)
 *   [96..191]  residual h3 - per-patient template
 *   [192..215] scalars: cols 0..12, 21..25, 32..37 of isoline.features.SCAL_NAMES
 *
 * Assumes arithmetic right shift of negative signed integers (true for GCC/Clang and arm-none-eabi-gcc),
 * which equals Python's floor shift.
 */
#ifndef ISOLINE_CORE_H
#define ISOLINE_CORE_H
#include <stdint.h>

#define ISO_FS 360
#define ISO_WIN 256
#define ISO_PRE 96
#define ISO_NBINS 32
#define ISO_BIN 8
#define ISO_RRQ 32
#define ISO_HIST_CLIP 255
#define ISO_K 4
#define ISO_NFEAT 216
#define ISO_NAF 10

typedef struct {
    /* leaky integrator */
    int32_t acc_state;
    /* beat-feature state (see features.py) */
    int primed_rr, sp_primed, tmpl_init, cen_init;
    int64_t rr_state, rrl_state, irl_state, rrv_state, irr_state, sp_state, dmin_state;
    int32_t prev_pre, prev_pre2, prev_rrr, prem_bits;
    int64_t tmpl8[3][ISO_NBINS];
    int64_t cen8[ISO_K][3][ISO_NBINS];
    int64_t cnt[ISO_K], crr[ISO_K];
    /* AF detector: previous RR intervals in seconds (16 + current) */
    double rr_hist[16];      /* previous RR intervals, seconds, numpy-identical arithmetic */
    int n_rr;
} iso_feat_state;

typedef struct {
    int64_t peak, steep, rr_avg;
    int64_t last;
    int n_beats;
} iso_qrs_state;

void iso_feat_init(iso_feat_state *s);
void iso_qrs_init(iso_qrs_state *s);

/* Leaky integrator for one slot (net = up - dn). Returns the signed 8-bit output. */
int32_t iso_leaky_step(iso_feat_state *s, int32_t net);

/* Offline-style API used by the verification harness and by the streaming firmware:
 * cu/cd are prefix sums of up/down counts (length n+1), acc the leaky-integrator output (length n),
 * r_prev / r_cur the previous and current beat positions (samples). Updates the state for every beat (as the hardware
 * must) and, when the beat window lies inside [0, n), writes the 216 features to out and returns 1; otherwise 0. */
int iso_beat_features(iso_feat_state *s, const int64_t *cu, const int64_t *cd, const int16_t *acc, int64_t n,
                      int64_t r_prev, int64_t r_cur, int32_t *out);

/* RR-only AF features from beat sample positions (previous 16 RR). Returns 1 if valid (>= 8 previous intervals). */
int iso_af_features(iso_feat_state *s, int64_t r_prev, int64_t r_cur, float *out);

/* Event-based beat detector over a whole buffer (mirrors isoline.qrs.detect). up/dn per slot, n slots.
 * Writes beat sample indices to beats (capacity max_beats), returns the number of beats. */
int iso_qrs_detect(const int16_t *up, const int16_t *dn, int64_t n, int64_t *beats, int max_beats);

#endif
