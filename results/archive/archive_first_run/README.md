# Archive: first run (superseded)

These files come from the first end-to-end run and are kept for transparency.
**Do not cite them as results.**

* `sweep_deltav.*` was computed on DS2 (test patients) and the level-crossing step
  dV = 0.1 mV was picked by looking at it. That is test-set information leaking into a
  design choice.
* `train_report.json` / `student.pt` inherit that dV choice and used the older feature set.

The choice was re-derived using DS1 only (`../select_delta.csv`, grouped CV, sparsity > 80%) and
agrees (0.1 mV). The current, clean results are `../train_final_v2.json`.
