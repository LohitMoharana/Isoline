"""Download the four benchmark datasets from the official PhysioNet servers.

Run this on a machine with internet access (it is NOT reachable from the
Claude sandbox). Example:

    python scripts/data/download_data.py --only mitdb

Datasets (all on https://physionet.org):
  mitdb          MIT-BIH Arrhythmia Database          ~100 MB   (needed first)
  svdb           MIT-BIH Supraventricular Database    ~ 55 MB   (extra TRAINING data for S)
  edb            European ST-T Database               ~ 90 MB
  ptbdb          PTB Diagnostic ECG Database          ~ 1.7 GB
  challenge2017  PhysioNet/CinC 2017 AF challenge     ~ 100 MB (training set)
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

CHALLENGE_URL = "https://physionet.org/files/challenge-2017/1.0.0/training2017.zip"
WFDB_DBS = {"mitdb": "mitdb", "edb": "edb", "ptbdb": "ptbdb", "svdb": "svdb"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "data"))
    ap.add_argument("--only", nargs="*", default=["mitdb"],
                    choices=list(WFDB_DBS) + ["challenge2017"],
                    help="datasets to fetch (default: mitdb)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name in args.only:
        dest = out / name
        if name in WFDB_DBS:
            import wfdb
            print(f"[{name}] downloading to {dest} ...")
            wfdb.dl_database(WFDB_DBS[name], str(dest))
        else:
            dest.mkdir(parents=True, exist_ok=True)
            zpath = dest / "training2017.zip"
            if not zpath.exists():
                print(f"[{name}] downloading {CHALLENGE_URL}")
                urllib.request.urlretrieve(CHALLENGE_URL, zpath)
            print(f"[{name}] extracting ...")
            with zipfile.ZipFile(zpath) as z:
                z.extractall(dest)
        print(f"[{name}] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
