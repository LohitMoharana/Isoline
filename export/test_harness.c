
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "isoline_model.h"
int main(int argc, char **argv) {
    FILE *fb = fopen(argv[1], "rb"), *fa = fopen(argv[2], "rb"), *fo = fopen(argv[3], "w");
    int n = atoi(argv[4]), nb = atoi(argv[5]), na = atoi(argv[6]), use_q = atoi(argv[7]);
    int32_t *xb = malloc(sizeof(int32_t) * nb); float *xa = malloc(sizeof(float) * (na + 1));
    for (int i = 0; i < n; i++) {
        if (fread(xb, sizeof(int32_t), nb, fb) != (size_t)nb) return 1;
        if (fread(xa, sizeof(float), na + 1, fa) != (size_t)(na + 1)) return 1;  /* last value: valid flag */
        fprintf(fo, "%d\n", isoline_classify(xb, xa[na] > 0.5f ? xa : NULL, use_q));
    }
    fclose(fo); return 0;
}
