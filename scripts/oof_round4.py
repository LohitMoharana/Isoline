"""Save round-4 DS1 leave-one-patient-out predictions (probabilities) and SVDB external predictions, for dev use."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isoline import data  # noqa: E402
from isoline.config import AFEConfig  # noqa: E402
from isoline.pipeline import build_dataset  # noqa: E402
import scripts.final_gbt as F  # noqa: E402

D, INC, SV, C, OUT = sys.argv[1:6]
cfg = replace(AFEConfig(), delta_mv=0.1)
ds1, _ = data.get_splits(D)
d1 = build_dataset(ds1, D, cfg, cache_dir=C, verbose=False)
e = build_dataset(sorted(p.stem for p in Path(INC).glob("*.hea")), INC, cfg, cache_dir=C, verbose=False, lead="II")
X1, y1, g1 = F.X_of(d1), d1["y"].astype(int), d1["rec"]
Xe, ye = F.X_of(e), e["y"].astype(int)
P1 = np.zeros((len(y1), 4))
for tr, te in GroupKFold(len(np.unique(g1))).split(X1, y1, g1):
    P1[te] = F.fit(X1[tr], y1[tr], Xe, ye, 0).predict_proba(X1[te])
v = build_dataset(sorted(p.stem for p in Path(SV).glob("*.hea")), SV, cfg, cache_dir=C, verbose=False, lead="ECG1")
Pv = F.fit(X1, y1, Xe, ye, 0).predict_proba(F.X_of(v))
np.savez(OUT, P1=P1, y1=y1, g1=g1, r1=d1["r"], Pv=Pv, yv=v["y"].astype(int), gv=v["rec"], rv=v["r"])
print("saved", OUT)
