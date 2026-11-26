#include "isoline_stream.h"
#include <string.h>

#define M (ISO_RB - 1)
#define Q_WIN 12
#define Q_REFRACT 72
#define Q_SEARCH 36
#define Q_LOOKBACK 18
#define Q_THR_MIN 3
#define Q_THR_SHIFT 2
#define Q_STEEP_WIN 4
#define Q_STEEP_SHIFT 1
#define POST (ISO_WIN - ISO_PRE)    /* 160 slots after R */

static inline int64_t imax(int64_t a, int64_t b) { return a > b ? a : b; }
static inline int64_t imin(int64_t a, int64_t b) { return a < b ? a : b; }
static inline int64_t iabs(int64_t a) { return a < 0 ? -a : a; }

void iso_stream_init(iso_stream *s) {
    memset(s, 0, sizeof(*s));
    iso_qrs_init(&s->qs); iso_feat_init(&s->fs); iso_feat_init(&s->as);
}

/* one iteration of isoline.qrs.detect's loop at position t_det (needs slots up to t_det + SEARCH - 1) */
static void detect_step(iso_stream *s) {
    int64_t t = s->t_det;
    iso_qrs_state *st = &s->qs;
    int64_t thr = imax(Q_THR_MIN, (st->peak >> 3) >> Q_THR_SHIFT);
    if (s->n_det && t - st->last > (13 * (st->rr_avg >> 3)) / 8) thr = imax(Q_THR_MIN, thr >> 1);
    if (t - st->last >= Q_REFRACT && s->E[t & M] >= thr) {
        int64_t s0 = imax(t - Q_LOOKBACK, 0), s1 = t + Q_SEARCH;
        int64_t base = s->q[s0 & M], best = s0, bv = -1;
        for (int64_t i = s0; i < s1; i++) { int64_t v = iabs(s->q[i & M] - base); if (v > bv) { bv = v; best = i; } }
        int64_t r = best;
        if (r - st->last < Q_REFRACT) { s->t_det = t + 1; return; }
        int64_t spk = 0, epk = 0;
        for (int64_t i = t; i < s1; i++) { if (s->S[i & M] > spk) spk = s->S[i & M]; if (s->E[i & M] > epk) epk = s->E[i & M]; }
        if (spk < imax(2, (st->steep >> 3) >> Q_STEEP_SHIFT)) { s->t_det = t + 1; return; }
        st->steep = st->steep - (st->steep >> 3) + spk;
        st->peak = st->peak - (st->peak >> 3) + epk;
        if (s->n_det) st->rr_avg = st->rr_avg - (st->rr_avg >> 3) + (r - st->last);
        s->n_det++; st->last = r;
        if (s->n_pending < ISO_PENDING) s->pending[s->n_pending++] = r;
        s->t_det = imax(r + Q_REFRACT, t + 1);
        return;
    }
    s->t_det = t + 1;
}

static void beat_out(iso_stream *s, int64_t r, iso_beat *b) {
    memset(b, 0, sizeof(*b));
    b->r = r;
    if (!s->have_last) {                       /* the first beat only primes RR state (as in the offline loop) */
        s->last_r = r; s->have_last = 1; b->have_feat = -1; return;
    }
    int64_t st = r - ISO_PRE;
    if (st >= 0) {                             /* copy the window [st, st+256] out of the ring and shift indices */
        int64_t cu[ISO_WIN + 1], cd[ISO_WIN + 1]; int16_t acc[ISO_WIN];
        for (int i = 0; i <= ISO_WIN; i++) {      /* rebase to the window start: exact modular differences */
            cu[i] = (int64_t)(uint32_t)(s->cu[(st + i) & M] - s->cu[st & M]);
            cd[i] = (int64_t)(uint32_t)(s->cd[(st + i) & M] - s->cd[st & M]);
        }
        for (int i = 0; i < ISO_WIN; i++) acc[i] = s->acc[(st + i) & M];
        b->have_feat = iso_beat_features(&s->fs, cu, cd, acc, ISO_WIN, s->last_r - st, r - st, b->feat);
    } else {
        int64_t dummy = 0; int16_t a0 = 0;
        b->have_feat = iso_beat_features(&s->fs, &dummy, &dummy, &a0, 0, s->last_r, r, b->feat);
    }
    b->af_valid = iso_af_features(&s->as, s->last_r, r, b->af);
    s->last_r = r;
}

int iso_stream_push(iso_stream *s, int up, int dn, iso_beat *out, int max_out) {
    int64_t t = s->T;
    /* prefix sums are "events before slot t": store cu for index t before adding this slot */
    s->cu[t & M] = (uint32_t)s->run_cu; s->cd[t & M] = (uint32_t)s->run_cd;
    s->up[t & M] = (int16_t)up; s->dn[t & M] = (int16_t)dn;
    s->acc[t & M] = (int16_t)iso_leaky_step(&s->fs, up - dn);
    s->run_cu += up; s->run_cd += dn;
    s->run_E += up + dn; if (t >= Q_WIN) s->run_E -= s->up[(t - Q_WIN) & M] + s->dn[(t - Q_WIN) & M];
    s->run_S += up + dn; if (t >= Q_STEEP_WIN) s->run_S -= s->up[(t - Q_STEEP_WIN) & M] + s->dn[(t - Q_STEEP_WIN) & M];
    s->run_q += up - dn;
    s->E[t & M] = (int16_t)s->run_E; s->S[t & M] = (int16_t)s->run_S; s->q[t & M] = (int32_t)s->run_q;
    s->T = t + 1;
    s->cu[s->T & M] = (uint32_t)s->run_cu; s->cd[s->T & M] = (uint32_t)s->run_cd;      /* end-exclusive prefix for the newest slot */
    while (s->t_det + Q_SEARCH <= s->T) detect_step(s);
    int n_out = 0;
    while (s->n_pending && s->pending[0] + POST <= s->T && n_out < max_out) {
        beat_out(s, s->pending[0], &out[n_out]);
        if (out[n_out].have_feat != -1) n_out++;
        memmove(s->pending, s->pending + 1, sizeof(int64_t) * (--s->n_pending));
    }
    return n_out;
}
