# First-Principles Physics & Collective-Kinematics Possession Tracker

Self-calibrated physical possession and dead-ball state machine (`solution/touch_possession.py`) coupled with a self-supervised relational team & goalkeeper classifier (`solution/player_team_classifier.py`) and an optional unsupervised main-referee pose prior (`solution/referee_pose_detector.py`). Zero human labels from the provided match are used.

For the full challenge technical report (approach, assets used & dropped, external datasets, possession rules, limitations, and multi-dataset benchmarks), see **[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)**.

## Pipeline

1. `solution/parse_atd.py` parses `data/tactical/BAR-ATH_tactical_atd.txt` into `cache/tactical_atd.npz` (`199,319` frames at 25 Hz).
2. `solution/player_team_classifier.py` fits a self-supervised 15-feature relational spatial-graph ensemble strictly outside `[11185, 18685)` to correct tracking identity switches (`99.86%` 5-fold CV accuracy).
3. `solution/touch_possession.py` self-calibrates kinematic velocity, acceleration, and collective slowdown quantiles on unlabeled tracking (`exclude_interval=(11185, 18685)`), suppresses `NaN`-interpolation boundary kinks, and decodes `H`, `A`, `D`, and `X` states using co-moving dribble control, 2D pass fly-by rejection, contested aerial parallax resolution, and FIFA penalty-area restart guards.

## Reproduce

```sh
pip install -r requirements.txt -r requirements-solution.txt
python3 solution/touch_possession.py --output my_submission.csv
python3 evaluation/validate_submission.py my_submission.csv
python3 evaluation/run_evaluation.py --gt public
```

| File | SHA-256 |
| --- | --- |
| `my_submission.csv` | `40801165dbd96516d6ec29b54db66cac1ed9fd35575a5e5b48a0ebb906b31caf` |
| `solution/touch_possession.py` | `f90adc9d20c0df79cd8ad220246a771166e5a6cd87dc03b730540efbb321b007` |
| `solution/player_team_classifier.py` | `cf07dfdcb871e5c2de2f8b803cdfd41904cff88a00031d2499ac699d5165b294` |
| `solution/referee_pose_detector.py` | `2d1fd3aeda90208d8aca21d6784562ec92d012efdfd6a3bb442bdb22dcda98ac` |
| `solution/whistle_detector.py` | `9bc4edbda1763cba331fc3de0258a50f2206c80cdbd6f49debb440d2127ce0a9` |
| `solution/parse_atd.py` | `d479424a8cd6c067fdafb7a8fe3a4e4df18f1e196315eb2c2cb032381f0fda5d` |
