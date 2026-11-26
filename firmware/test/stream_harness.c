/* Streams one record slot by slot through iso_stream_push and writes every emitted beat (r, features, AF). */
#include <stdio.h>
#include <stdlib.h>
#include "../core/isoline_stream.h"

int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb"); int64_t n, nr;
    if (fread(&n, 8, 1, f) != 1 || fread(&nr, 8, 1, f) != 1) return 1;
    int16_t *up = malloc(2 * n), *dn = malloc(2 * n);
    if (fread(up, 2, n, f) != (size_t)n || fread(dn, 2, n, f) != (size_t)n) return 1;
    fclose(f);
    static iso_stream s; iso_stream_init(&s);
    iso_beat b[4];
    FILE *o = fopen(argv[2], "wb");
    for (int64_t t = 0; t < n; t++) {
        int k = iso_stream_push(&s, up[t], dn[t], b, 4);
        for (int j = 0; j < k; j++) {
            int64_t hdr[3] = {b[j].r, b[j].have_feat, b[j].af_valid};
            fwrite(hdr, 8, 3, o); fwrite(b[j].feat, 4, ISO_NFEAT, o); fwrite(b[j].af, 4, ISO_NAF, o);
        }
    }
    fclose(o); return 0;
}
