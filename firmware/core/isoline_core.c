/* Isoline core, see isoline_core.h. Each block cites the Python line it mirrors (isoline/features.py, isoline/qrs.py,
 * scripts/final/af_detector.py). */
#include "isoline_core.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define ACC_SHIFT 4
#define ACC_FRAC 4
#define GAIN_T_LOG2 5

static inline int64_t sm_shift(int64_t a, int k) { return a >= 0 ? (a >> k) : -((-a) >> k); }
static inline int64_t imin(int64_t a, int64_t b) { return a < b ? a : b; }
static inline int64_t imax(int64_t a, int64_t b) { return a > b ? a : b; }
static inline int64_t iabs(int64_t a) { return a < 0 ? -a : a; }
static inline int bitlen(int64_t v) { int b = 0; while (v > 0) { b++; v >>= 1; } return b; }

void iso_feat_init(iso_feat_state *s) {
    memset(s, 0, sizeof(*s));
    s->prev_rrr = 32;
    for (int k = 0; k < ISO_K; k++) s->crr[k] = 8 * 32;
}

void iso_qrs_init(iso_qrs_state *s) { s->peak = 8 * 16; s->steep = 8 * 6; s->rr_avg = 8 * 288; s->last = -72; s->n_beats = 0; }

int32_t iso_leaky_step(iso_feat_state *s, int32_t net) {          /* features._leaky_integrator */
    const int32_t slo = -(128 << ACC_FRAC), shi = ((127 + 1) << ACC_FRAC) - 1;
    int32_t st = s->acc_state;
    int32_t dec = st >= 0 ? (st >> ACC_SHIFT) : -((-st) >> ACC_SHIFT);
    st = st - dec + net * (1 << ACC_FRAC);
    if (st > shi) st = shi; else if (st < slo) st = slo;
    s->acc_state = st;
    return st >= 0 ? (st >> ACC_FRAC) : -((-st) >> ACC_FRAC);
}

static int64_t rr_counts(int64_t d) { return imin((8 * d + 45) / 90, 65535); } /* round(d*32/360), no ties */

int iso_beat_features(iso_feat_state *s, const int64_t *cu, const int64_t *cd, const int16_t *acc, int64_t n,
                      int64_t r_prev, int64_t r_cur, int32_t *out) {
    int64_t pre = rr_counts(r_cur - r_prev);
    if (!s->primed_rr) {
        s->rr_state = 8 * pre; s->rrl_state = 64 * pre; s->rrv_state = 256 * pre;
        s->prev_pre = (int32_t)pre; s->prev_pre2 = (int32_t)pre; s->primed_rr = 1;
    }
    int64_t rr_ewma_prev = s->rr_state >> 3, rrl = s->rrl_state >> 6;
    int64_t delta = pre - s->prev_pre;
    int64_t rr_ratio = imin((pre * ISO_RRQ) / imax(rr_ewma_prev, 1), 255);
    int64_t rr_ratio_prev = imin((pre * ISO_RRQ) / imax(s->prev_pre, 1), 255);
    int64_t st = r_cur - ISO_PRE, en = st + ISO_WIN;
    int have = (st >= 0 && en <= n);
    if (have) {
        int64_t h3[3][ISO_NBINS], tmpl[3][ISO_NBINS], r3[3][ISO_NBINS];
        int64_t hu[ISO_NBINS], hd[ISO_NBINS], ha[ISO_NBINS];
        for (int b = 0; b < ISO_NBINS; b++) {
            int64_t e0 = st + ISO_BIN * b, e1 = e0 + ISO_BIN;
            hu[b] = imin(cu[e1] - cu[e0], ISO_HIST_CLIP);
            hd[b] = imin(cd[e1] - cd[e0], ISO_HIST_CLIP);
            ha[b] = acc[e1 - 1];
        }
        int64_t beat_spikes = (cu[en] - cu[st]) + (cd[en] - cd[st]);
        if (!s->sp_primed) { s->sp_state = 8 * beat_spikes; s->sp_primed = 1; }
        int64_t sp_prev = s->sp_state >> 3;
        s->sp_state = s->sp_state - (s->sp_state >> 3) + beat_spikes;
        int k = (int)imin(imax(bitlen(sp_prev) - (GAIN_T_LOG2 + 1), 0), 6);
        for (int b = 0; b < ISO_NBINS; b++) {
            h3[0][b] = sm_shift(hu[b], k); h3[1][b] = sm_shift(hd[b], k); h3[2][b] = sm_shift(ha[b], k);
        }
        int64_t sp_ratio = imin((beat_spikes * 32) / imax(sp_prev, 1), 255);
        /* per-patient template */
        if (!s->tmpl_init) {
            for (int c = 0; c < 3; c++) for (int b = 0; b < ISO_NBINS; b++) { tmpl[c][b] = h3[c][b]; s->tmpl8[c][b] = 8 * h3[c][b]; }
            s->tmpl_init = 1;
        } else {
            for (int c = 0; c < 3; c++) for (int b = 0; b < ISO_NBINS; b++) tmpl[c][b] = sm_shift(s->tmpl8[c][b], 3);
            if (rr_ratio >= 24 && rr_ratio <= 40)
                for (int c = 0; c < 3; c++) for (int b = 0; b < ISO_NBINS; b++) s->tmpl8[c][b] += h3[c][b] - tmpl[c][b];
        }
        for (int c = 0; c < 3; c++) for (int b = 0; b < ISO_NBINS; b++) r3[c][b] = h3[c][b] - tmpl[c][b];
        /* online clustering (leader algorithm) */
        if (!s->cen_init) {
            memset(s->cen8, 0, sizeof(s->cen8));
            for (int c = 0; c < 3; c++) for (int b = 0; b < ISO_NBINS; b++) s->cen8[0][c][b] = 8 * h3[c][b];
            s->cnt[0] = 16 * 4; s->cen_init = 1;
        }
        int64_t dk[ISO_K]; int jn = 0;
        for (int q = 0; q < ISO_K; q++) {
            int64_t d = 0;
            for (int c = 0; c < 3; c++) for (int b = 0; b < ISO_NBINS; b++) d += iabs((s->cen8[q][c][b] >> 3) - h3[c][b]);
            dk[q] = s->cnt[q] > 0 ? d : ((int64_t)1 << 30);
            if (dk[q] < dk[jn]) jn = q;
        }
        int64_t dn_ = imin(dk[jn], 65535), tot = 0, active = 0, rank = 0;
        for (int q = 0; q < ISO_K; q++) { tot += s->cnt[q]; active += s->cnt[q] > 0; }
        for (int q = 0; q < ISO_K; q++) rank += s->cnt[q] > s->cnt[jn];
        int64_t pop = imin((s->cnt[jn] * 32) / imax(tot, 1), 32);
        int64_t c_rr = s->crr[jn] >> 3;
        int64_t dthr = imax(2 * (s->dmin_state >> 3), 16);
        int64_t clu_dist_ratio = imin((dn_ * 32) / imax(s->dmin_state >> 3, 1), 255);
        int64_t clu_feats[6] = {dn_, pop, rank, imin(c_rr, 255), active, clu_dist_ratio};
        if (dn_ > dthr && active < ISO_K) {
            int z = 0; while (s->cnt[z] != 0) z++;
            jn = z;
            for (int c = 0; c < 3; c++) for (int b = 0; b < ISO_NBINS; b++) s->cen8[jn][c][b] = 8 * h3[c][b];
            s->crr[jn] = 8 * rr_ratio;
        } else if (dn_ > dthr) {
            int z = 0; for (int q = 1; q < ISO_K; q++) if (s->cnt[q] < s->cnt[z]) z = q;
            jn = z;
            for (int c = 0; c < 3; c++) for (int b = 0; b < ISO_NBINS; b++) s->cen8[jn][c][b] = 8 * h3[c][b];
            s->crr[jn] = 8 * rr_ratio; s->cnt[jn] = 0;
        } else {
            for (int c = 0; c < 3; c++) for (int b = 0; b < ISO_NBINS; b++)
                s->cen8[jn][c][b] = s->cen8[jn][c][b] - (s->cen8[jn][c][b] >> 3) + h3[c][b];
            s->crr[jn] = s->crr[jn] - (s->crr[jn] >> 3) + rr_ratio;
        }
        for (int q = 0; q < ISO_K; q++) s->cnt[q] -= s->cnt[q] >> 4;
        s->cnt[jn] += 16;
        s->dmin_state = s->dmin_state - (s->dmin_state >> 3) + imin(dn_, 4095);
        int64_t dch[3] = {0, 0, 0};
        for (int c = 0; c < 3; c++) { for (int b = 0; b < ISO_NBINS; b++) dch[c] += iabs(r3[c][b]); dch[c] = imin(dch[c], 65535); }
        /* ---- write the 216-value vector ---- */
        int o = 0;
        for (int c = 0; c < 3; c++) for (int b = 0; b < ISO_NBINS; b++) out[o++] = (int32_t)h3[c][b];
        for (int c = 0; c < 3; c++) for (int b = 0; b < ISO_NBINS; b++) out[o++] = (int32_t)r3[c][b];
        int64_t scal0[13] = {pre, rr_ewma_prev, delta, beat_spikes, sp_prev, rr_ratio, sp_ratio, s->prev_pre,
                             rr_ratio_prev, dch[0], dch[1], dch[2], dch[0] + dch[1] + dch[2]};
        for (int i = 0; i < 13; i++) out[o++] = (int32_t)scal0[i];
        int pc = 0; for (int b = 0; b < 8; b++) pc += (s->prem_bits >> b) & 1;
        int64_t longf[5] = {rrl, imin((pre * ISO_RRQ) / imax(rrl, 1), 255), imin((rr_ewma_prev * ISO_RRQ) / imax(rrl, 1), 255),
                            imin(((s->irl_state >> 6) * ISO_RRQ) / imax(rrl, 1), 255), pc};
        for (int i = 0; i < 5; i++) out[o++] = (int32_t)longf[i];
        for (int i = 0; i < 6; i++) out[o++] = (int32_t)clu_feats[i];
    }
    /* per-beat state updates for every beat (features.py end of loop) */
    s->rr_state = s->rr_state - (s->rr_state >> 3) + pre;
    s->irr_state = s->irr_state - (s->irr_state >> 3) + iabs(pre - s->prev_pre);
    s->rrl_state = s->rrl_state - (s->rrl_state >> 6) + pre;
    s->irl_state = s->irl_state - (s->irl_state >> 6) + iabs(pre - s->prev_pre);
    s->rrv_state = s->rrv_state - (s->rrv_state >> 8) + pre;
    s->prem_bits = ((s->prem_bits << 1) | (rr_ratio < 28)) & 0xFF;
    s->prev_rrr = (int32_t)rr_ratio;
    s->prev_pre2 = s->prev_pre;
    s->prev_pre = (int32_t)pre;
    return have;
}

/* scripts/af_detector.rr_features, reproduced in double precision exactly as numpy computes it:
 * rr = diff(r / 360.0) (so equal-length intervals can differ by ~1e-16, which the trained detector has seen),
 * h = the previous (up to) 16 intervals, cur = current interval / median(h). */
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static double median_d(double *v, int n) { qsort(v, n, sizeof(double), cmpd); return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]); }
static double pct_d(double *v, int n, double p) {
    qsort(v, n, sizeof(double), cmpd); double idx = p / 100.0 * (n - 1); int lo = (int)idx; double fr = idx - lo;
    return lo + 1 < n ? v[lo] + (v[lo + 1] - v[lo]) * fr : v[lo];
}

int iso_af_features(iso_feat_state *s, int64_t r_prev, int64_t r_cur, float *out) {
    double rr_cur = (double)r_cur / 360.0 - (double)r_prev / 360.0;
    int valid = 0, n = s->n_rr, m = n < 16 ? n : 16;
    if (m >= 8) {
        double h[16], t[16], d[16];
        for (int j = 0; j < m; j++) h[j] = s->rr_hist[n - m + j];
        memcpy(t, h, sizeof(double) * m); double med = median_d(t, m);
        double mean = 0; for (int j = 0; j < m; j++) mean += h[j]; mean /= m;
        double sd = 0; for (int j = 0; j < m; j++) sd += (h[j] - mean) * (h[j] - mean); sd = sqrt(sd / m);
        for (int j = 0; j < m - 1; j++) d[j] = fabs(h[j + 1] - h[j]);
        double dmean = 0; for (int j = 0; j < m - 1; j++) dmean += d[j]; dmean /= (m - 1);
        memcpy(t, d, sizeof(double) * (m - 1)); double dmed = median_d(t, m - 1);
        double dom = 0, fs = 0, fl = 0, tp = 0, sl = 0, domp = 0;
        for (int j = 0; j < m; j++) { dom += fabs(h[j] - med) < 0.1 * med; fs += h[j] < 0.85 * med; fl += h[j] > 1.15 * med; }
        for (int j = 1; j < m - 1; j++) tp += (h[j] - h[j - 1]) * (h[j] - h[j + 1]) > 0;
        memcpy(t, h, sizeof(double) * m); double p75 = pct_d(t, m, 75.0);
        for (int j = 0; j < m; j++) domp += fabs(h[j] - p75) < 0.1 * p75;
        for (int j = 0; j < m - 1; j++) sl += (h[j] < 0.9 * med) && (h[j + 1] > 1.05 * med);
        double o[ISO_NAF] = {dmed / med, dmean / mean, dom / m, fs / m, fl / m, sd / mean, tp / (m - 2), rr_cur / med,
                             domp / m, sl / (m - 1)};
        for (int k = 0; k < ISO_NAF; k++) out[k] = (float)o[k];
        valid = 1;
    }
    if (s->n_rr == 16) { memmove(s->rr_hist, s->rr_hist + 1, sizeof(double) * 15); s->n_rr = 15; }
    s->rr_hist[s->n_rr++] = rr_cur;
    return valid;
}

/* isoline.qrs.detect */
#define Q_WIN 12
#define Q_REFRACT 72
#define Q_SEARCH 36
#define Q_LOOKBACK 18
#define Q_THR_MIN 3
#define Q_THR_SHIFT 2
#define Q_STEEP_WIN 4
#define Q_STEEP_SHIFT 1

int iso_qrs_detect(const int16_t *up, const int16_t *dn, int64_t n, int64_t *beats, int max_beats) {
    int64_t *E = malloc(sizeof(int64_t) * n), *S = malloc(sizeof(int64_t) * n), *q = malloc(sizeof(int64_t) * n);
    int64_t e = 0, sw = 0, acc = 0;
    for (int64_t t = 0; t < n; t++) {
        int64_t a = up[t] + dn[t];
        e += a; if (t >= Q_WIN) e -= up[t - Q_WIN] + dn[t - Q_WIN];
        sw += a; if (t >= Q_STEEP_WIN) sw -= up[t - Q_STEEP_WIN] + dn[t - Q_STEEP_WIN];
        acc += up[t] - dn[t];
        E[t] = e; S[t] = sw; q[t] = acc;
    }
    iso_qrs_state st; iso_qrs_init(&st);
    int nb = 0; int64_t t = 0;
    while (t < n) {
        int64_t thr = imax(Q_THR_MIN, (st.peak >> 3) >> Q_THR_SHIFT);
        if (nb && t - st.last > (13 * (st.rr_avg >> 3)) / 8) thr = imax(Q_THR_MIN, thr >> 1);
        if (t - st.last >= Q_REFRACT && E[t] >= thr) {
            int64_t s0 = imax(t - Q_LOOKBACK, 0), s1 = imin(t + Q_SEARCH, n);
            int64_t base = q[s0], best = s0, bv = -1;
            for (int64_t i = s0; i < s1; i++) { int64_t v = iabs(q[i] - base); if (v > bv) { bv = v; best = i; } }
            int64_t r = best;
            if (r - st.last < Q_REFRACT) { t++; continue; }
            int64_t spk = 0, epk = 0;
            for (int64_t i = t; i < s1; i++) { if (S[i] > spk) spk = S[i]; if (E[i] > epk) epk = E[i]; }
            if (spk < imax(2, (st.steep >> 3) >> Q_STEEP_SHIFT)) { t++; continue; }
            st.steep = st.steep - (st.steep >> 3) + spk;
            st.peak = st.peak - (st.peak >> 3) + epk;
            if (nb) st.rr_avg = st.rr_avg - (st.rr_avg >> 3) + (r - st.last);
            if (nb < max_beats) beats[nb] = r;
            nb++; st.last = r;
            t = imax(r + Q_REFRACT, t + 1);
            continue;
        }
        t++;
    }
    free(E); free(S); free(q);
    return nb < max_beats ? nb : max_beats;
}
