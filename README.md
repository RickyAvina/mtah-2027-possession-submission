# More Than A Hack 2027 — First-Principles Physics & Collective-Kinematics Possession Tracker

**Phase 1 Submission Repository** for the [More Than A Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027) challenge by [Metrica Sports](https://metrica-sports.com), [FC Barcelona](https://www.fcbarcelona.com), and [Mobile World Capital Barcelona](https://www.mwcbarcelona.com).

- **Full Technical Report**: **[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)**
- **Primary Model Entrypoint**: `solution/touch_possession.py`
- **Submission Checksum (`my_submission.csv`)**: `40801165dbd96516d6ec29b54db66cac1ed9fd35575a5e5b48a0ebb906b31caf` (`199,319` rows: `158,857` graded frames + `40,462` ungraded `X` frames)

---

## Quickstart: Reproduce `my_submission.csv`

Requires Python 3.10+ and the gated competition data bundle unpacked under `data/` (or pointed to by `MTAH_DATA_DIR`).

```bash
# 1. Create virtual environment and install pinned dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-solution.txt

# 2. Generate the exact 199,319-row submission file (automatically builds caches if missing)
python3 solution/touch_possession.py --output my_submission.csv

# 3. Validate format and row alignment with the official validator
python3 evaluation/validate_submission.py my_submission.csv

# 4. Score against the public 5-minute ground truth
python3 evaluation/run_evaluation.py --gt public
```

---

## Results Summary

### 1. Public Validation Slice (`BAR-ATH [11185, 18685)`)
All self-supervised team classification and empirical velocity/acceleration quantile calibrations strictly exclude `[11185, 18685)`.

| Submission | `spellF1` (IoU 0.3–0.7) | `macroF1` | Frame Accuracy | Possessions (`GT = 17`) | Transition F1 (`@25f`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`my_submission` (Ours — Physics Tracker)** | **`1.0000`** | **`0.9917`** | **`0.9911`** | **`17`** (`14.2 s`) | **`1.0000`** |
| `baseline_smooth125` | `0.4242` | `0.5812` | `0.7275` | `16` (`18.5 s`) | `0.1667` |
| `baseline_smooth50` | `0.3256` | `0.6207` | `0.7429` | `26` (`10.8 s`) | `0.0816` |
| `baseline_smooth25` | `0.2868` | `0.6200` | `0.7332` | `36` (`7.6 s`) | `0.2034` |
| `baseline_raw` | `0.0378` | `0.5972` | `0.6512` | `131` (`2.0 s`) | `0.1410` |

### 2. Multi-Dataset Generalization & 2D Pass Fly-By Stress Tests
Reproduce all cross-validation and multi-dataset experiments via:

```bash
python3 solution/cross_validate.py
python3 solution/multi_dataset_experiment.py
```

- **Non-Contact 2D Pass Fly-By Immunity**: **`100.0%`** real & synthetic fly-by rejection on `BAR-ATH` (`16/16` real events, `29/29` synthetic pass fly-overs); **`90.0–93.4%`** real fly-by rejection and **`99.6–100.0%`** synthetic immunity across `1,307` injected pass fly-overs on `Metrica Sports` (`Sample_Game_1..3`).
- **5-Fold Non-Public `BAR-ATH` Cross-Validation**: **`99.86%`** out-of-fold team recovery (`PlayerTeamClassifier`), **`100.0%`** goal-stoppage accuracy, **`93.10%`** set-piece stoppage recall.
- **External Benchmarks (`12` Full Matches, `1.53M` Frames)**:
  - **Metrica Sports (3 Matches, 25 Hz)**: `macroF1 = 0.8785`, `spellF1 = 0.6935` (`ProviderTouch`); `macroF1 = 0.8705` (`raw`).
  - **DFL Open Data (7 Bundesliga Matches, 25 Hz)**: `macroF1 = 0.9015` (`clean`), `macroF1 = 0.8702` (`missing-ball` with 17% ball dropout).
  - **SkillCorner Open Data (2 Matches, 10 Hz Broadcast Tracking)**: `macroF1 = 0.7429` (`extrapolated`), `macroF1 = 0.6881` (`broadcast-detected` with 55% missing ball).

---

## Repository Structure

- `solution/touch_possession.py` — Main self-calibrated physics & collective-kinematics possession state machine
- `solution/player_team_classifier.py` — Self-supervised 15-feature relational spatial-graph team & goalkeeper classifier
- `solution/whistle_detector.py` — Unsupervised `2.6–3.9 kHz` spectral referee whistle detector (`BAR-ATH_tactical.mp4` audio)
- `solution/referee_pose_detector.py` — Homography-guided untracked Main Referee detector & self-calibrated 2D pose prior (`yolov8n-pose.pt`)
- `solution/parse_atd.py` — Parser for `BAR-ATH_tactical_atd.txt` into `cache/tactical_atd.npz`
- `solution/cross_validate.py` — 5-fold non-public `BAR-ATH` CV, 3-fold `Metrica` LOGO-CV, and Real/Synthetic 2D Fly-By stress tests
- `solution/multi_dataset_experiment.py` — 15-match cross-provider benchmark (`BAR-ATH`, `Metrica`, `DFL`, `SkillCorner`)
- `TECHNICAL_REPORT.md` — Challenge technical report covering approach, assets used/dropped, external datasets, possession rules, limitations, and reproduction
- `evaluation/` & `starter/` — Official competition validation and scoring scripts ([CHALLENGE.md](CHALLENGE.md), [DATA.md](DATA.md), [EVALUATION.md](EVALUATION.md), [RULES.md](RULES.md), [SUBMISSION.md](SUBMISSION.md))
