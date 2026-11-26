/* Verification harness: reads one record's comparator events + annotated R positions, writes what the C core
 * computes (leaky integrator, 216-feature vectors, detected beats, AF features) for comparison with Python. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "../core/isoline_core.h"

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: harness in.bin out_prefix\n"); return 2; }
    FILE *f = fopen(argv[1], "rb"); int64_t n, nr;
    if (fread(&n, 8, 1, f) != 1 || fread(&nr, 8, 1, f) != 1) return 1;
    int16_t *up = malloc(2 * n), *dn = malloc(2 * n), *acc = malloc(2 * n); int64_t *r = malloc(8 * nr);
    if (fread(up, 2, n, f) != (size_t)n || fread(dn, 2, n, f) != (size_t)n || fread(r, 8, nr, f) != (size_t)nr) return 1;
    fclose(f);
    iso_feat_state st; iso_feat_init(&st);
    int64_t *cu = malloc(8 * (n + 1)), *cd = malloc(8 * (n + 1)); cu[0] = cd[0] = 0;
    for (int64_t t = 0; t < n; t++) { acc[t] = (int16_t)iso_leaky_step(&st, up[t] - dn[t]); cu[t + 1] = cu[t] + up[t]; cd[t + 1] = cd[t] + dn[t]; }
    char path[512];
    snprintf(path, sizeof path, "%s_acc.bin", argv[2]); f = fopen(path, "wb"); fwrite(acc, 2, n, f); fclose(f);
    snprintf(path, sizeof path, "%s_feat.bin", argv[2]); f = fopen(path, "wb");
    snprintf(path, sizeof path, "%s_af.bin", argv[2]); FILE *fa = fopen(path, "wb");
    int32_t feat[ISO_NFEAT]; float af[ISO_NAF];
    iso_feat_state st2; iso_feat_init(&st2);
    for (int64_t i = 1; i < nr; i++) {
        if (iso_beat_features(&st, cu, cd, acc, n, r[i - 1], r[i], feat)) { int64_t idx = i; fwrite(&idx, 8, 1, f); fwrite(feat, 4, ISO_NFEAT, f); }
    }
    for (int64_t i = 1; i < nr; i++) {                       /* AF features over the same beat sequence */
        int v = iso_af_features(&st2, r[i - 1], r[i], af);
        int64_t idx = v ? i : -i; fwrite(&idx, 8, 1, fa); fwrite(af, 4, ISO_NAF, fa);
    }
    fclose(f); fclose(fa);
    int64_t *beats = malloc(8 * (n / 50 + 10));
    int nb = iso_qrs_detect(up, dn, n, beats, (int)(n / 50 + 10));
    snprintf(path, sizeof path, "%s_qrs.bin", argv[2]); f = fopen(path, "wb"); fwrite(beats, 8, nb, f); fclose(f);
    return 0;
}
