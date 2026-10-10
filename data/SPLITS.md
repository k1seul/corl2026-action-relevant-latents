# Episode splits

All splits are lists of LIBERO episodes (`suite`, `file`, `demo`) over the five LIBERO suites (130 tasks).

| File | Used for |
|---|---|
| `splits_taskheld/train.json` | Probe and +ID training episodes from the 104 training tasks |
| `splits_taskheld/test_ood.json` | Held-out-task test pool: 26 tasks, 1,300 episodes |
| `splits_taskheld/test_indist.json` | Held-out episodes of the 104 training tasks |
| `splits/{train,test}.json` | Source episode-level split over all 130 tasks; `scripts/make_task_split.py` derives `splits_taskheld/` from it |
| `../runs/corl2026_rebuttal/splits/` | Copies used by the matched-control study, with SHA-256 files |

The absolute probe results in the paper (Table 1, Figures 1–3) use a fixed subset of these files: probes are fit on the first 400 training episodes (10 of the 104 training tasks) and evaluated on the first 200 `test_ood` episodes (4 of the 26 held-out tasks). The matched controls use all 26 held-out tasks.

## Changes from the files used in the experiments

The only change is `meta.datasets_dir`, which held a cluster path and now reads `datasets/libero`. The episode lists are unchanged. SHA-256 of the original files:

| File | Original SHA-256 |
|---|---|
| `runs/corl2026_rebuttal/splits/train.json` | `501ad00a68ee0d5d7cc50d2d7ca94160003ef6f9680d32f923dd3f7f858085f6` |
| `runs/corl2026_rebuttal/splits/test_ood.json` | `42f26dac66acbaf0c5a153f15605b09f9f9ea772c5647b7d20518874f2c769ca` (unchanged) |
| `runs/corl2026_rebuttal/splits/test_indist.json` | `0ea14388febd7f51a5508e3b3d41dbe7296e1d7bec021bc36a1e8ece40b08f05` |
