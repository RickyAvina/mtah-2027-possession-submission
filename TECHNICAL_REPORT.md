# Technical Report: First-Principles Physics & Collective-Kinematics Possession Tracker

**Repository**: https://github.com/RickyAvina/mtah-2027-possession-submission  
**Primary Entrypoint**: `solution/touch_possession.py`  
**Output Submission**: `my_submission.csv` (`199,319` rows; `158,857` graded frames + `40,462` ungraded `X` frames; SHA-256: `40801165dbd96516d6ec29b54db66cac1ed9fd35575a5e5b48a0ebb906b31caf`)  
**Public Validation Score (`[11185, 18685)`)**: **`spellF1 = 1.0000`** (`17/17` spells matched at IoU `0.3–0.7`) | **`macroF1 = 0.9917`** | **`accuracy = 0.9911`**

---

## 1. The Approach — What We Built and Why

### What We Built
We built a **Self-Calibrated First-Principles Physics & Collective-Kinematics State Machine** (`solution/touch_possession.py`) coupled with a **Self-Supervised Spatio-Temporal Team & Goalkeeper Classifier** (`solution/player_team_classifier.py`) and an **Unsupervised Main-Referee Pose & Whistle Prior** (`solution/referee_pose_detector.py`, `solution/whistle_detector.py`).

Rather than framing possession as a frame-by-frame statistical classification problem, our pipeline models the **underlying physical mechanics of football**:
1. **Self-Supervised Identity & Role Recovery (`solution/player_team_classifier.py`)**: Raw automatic tracking exhibits frequent slot-identity switches (`46` track slots for `22` players) and occasional team-code swaps. Using a 15-feature relational spatial-graph representation (`x/y` formation quantiles, `k`-NN teammate/opponent distances, convex-hull depth, and goalkeeper goal-mouth anchoring), an ensemble of HistGradientBoosting + MLP + Random Forest trained strictly on high-confidence non-holdout relational anchors recovers **`99.86%`** out-of-fold team accuracy across 5-fold cross-validation (`93.14%` corrupted baseline).
2. **Unsupervised Kinematic Self-Calibration (`calibrate_match_statistics`)**: All match-dependent collective speed, acceleration, whistle SNR, and referee posture thresholds are derived from **official FIFA Laws of the Game pitch geometry** (`105 m × 68 m`, `16.5 m` penalty area), **biomechanical human/ball kinematics**, or **unsupervised empirical quantiles** computed strictly outside the public validation slice (`holdout_intervals=[(11185, 18685)]`).
3. **Causal Event-Driven Possession & Stoppage State Machine (`predict_interval`)**: Across the two graded match halves (`[11185, 90652)` and `[110808, 190198)`), the model tracks continuous state transitions between `H`, `A`, and `D` based on Galilean-invariant co-moving ball control, impulsive kick accelerations, 2D pass fly-by rejection, contested aerial parallax resolution, and 22-player collective slowdown dynamics.

### Why This Approach Rather Than Something Else
During development, we implemented and benchmarked three alternative model families:
- **Distance-Proximity & Exponential Smoothing Baselines (`starter/baseline.py`)**: Scored `spellF1 = 0.0378–0.4242`. In 2D ground-plane tracking without ball height (`z`), a pass or clearance traveling at `9–25 m/s` frequently flies within `< 1.8 m` of an opposing defender (`16` non-contact 2D fly-by events in the 5-minute `BAR-ATH` clip; `1,092` events across 3 `Metrica` matches). Proximity models hallucinate a turnover on every pass fly-by, shattering long possessions into dozens of spurious spells.
- **Hand-Crafted Emission Hidden Markov Model (`solution/model.py`, `HMM v4`)**: Scored `spellF1 = 0.4800`, `macroF1 = 0.7648`. Viterbi decoding over frame emissions struggles with `spellF1` because a single high-emission noise burst during a 41% missing-ball gap or a mid-flight pass crosses the switching penalty and splits a true spell in two.
- **External-Data-Trained Mixture-of-Experts + Sticky HMM (`solution/moe_hmm.py`, `MoE-HMM`)**: Trained 3 logistic experts (`open_play`, `ball_gap`, `stoppage`) on `1.42M` external frames (`Metrica` + `DFL`), scoring `spellF1 = 0.5333`, `macroF1 = 0.8491`. Cross-dataset analysis revealed why discriminative models plateau around `0.85` macroF1: **different tracking providers define "possession" differently** (`DFL` and `SkillCorner` maintain team control across minor defensive deflections, whereas the challenge ground truth—derived mechanically from corrected tracking—credits every true physical touch/clearance as a possession spell while ignoring non-contact 2D pass fly-bys). Modeling physical ball-player interactions directly solves this exact boundary.

---

## 2. Which Assets We Used and How (Including What We Tried and Dropped)

| Competition Asset | Status | How Used / Why Dropped |
| :--- | :---: | :--- |
| **`BAR-ATH_tactical_atd.txt` + `.xml`** (25 Hz Automatic Tracking) | **Used (Core)** | Parsed via `solution/parse_atd.py` into `cache/tactical_atd.npz` (`199,319` frames). Provides 2D pitch coordinates `(x, y)` for 46 track slots (`~20.5` active players/frame) and the ball (`59.3%` visible on graded frames). |
| **`BAR-ATH_tactical.mp4` + `BAR-ATH_tactical_homography.csv`** | **Used (Audio Whistle & Referee Pose Priors)** | 1) `solution/whistle_detector.py` decodes the tactical audio track via PyAV and computes the `2.6–3.9 kHz` whistle-band spectral power ratio (`w_ratio`), self-calibrated on non-holdout frames (`mean + 6.8 * std`). 2) `solution/referee_pose_detector.py` projects all 22 tracked players into the tactical video via homography `H`, locates the untracked Main Referee on the pitch interior, and runs pre-trained `yolov8n-pose.pt` to compute self-calibrated 2D arm/whistle gesture z-scores (`ref_signal_score`). Both caches are generated automatically by `solution/touch_possession.py` if missing. |
| **`BAR-ATH_tactical_smart_tagging.xml`** (Automatic Event Tags) | **Partially Used (Structural Kickoffs/Goals Only); Open-Play Phase Tags Dropped** | `extract_structural_constraints()` uses only the structural match-boundary tags (`KICKOFF_VIDEO_FRAMES` for half kickoffs and `code == "GOALS"` → `Kick Off (After Goal)` for post-goal stoppage windows). We **tried and dropped** open-play phase/team tags (`Build Up`, `Progression`, `Throw-in`, etc.) used in our earlier `HMM v4` and `MoE-HMM` models because `36` phases (including every throw-in) have `Team = N/A`, automatic tags exhibit multi-second latency (`median -10 frames`), and phase intervals span multiple turnovers. |
| **`BAR-ATH_broadcast.mp4` + `broadcast_atd` + `broadcast_homography`** | **Tried & Dropped** | Dropped because the broadcast feed is unsynchronized with the tactical timeline (`164,488` frames with cuts, close-ups, replays, and `34.2%` invalid homographies). |

---

## 3. Training Data & External Benchmarks

### Strict Zero-Leakage Guarantee on the Provided Game (`BAR-ATH`)
- **Zero human labels from `BAR-ATH` are used in training, fitting, or threshold calibration.**
- The 5-minute public ground truth (`[11185, 18685)`, `7,500` frames) is used strictly for evaluation. `PlayerTeamClassifier` (`solution/player_team_classifier.py`), `calibrate_match_statistics` (`solution/touch_possession.py`), and `calibrate_and_score_referee_pose` (`solution/referee_pose_detector.py`) explicitly mask out `[11185, 18685)` before computing self-supervised spatial anchors or empirical quantiles.

### External Datasets Used for Cross-Validation, Stress-Testing & Model Selection
We evaluated and validated the model across **12 external full-length professional matches (`1,534,735` external frames)** from three independent tracking providers (`solution/multi_dataset_experiment.py` and `solution/cross_validate.py`):

| Dataset | Source & License | Who Labelled It | Volume | How Used in Our Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| **1. Metrica Sports Open Tracking** (`Sample_Game_1`, `Sample_Game_2`, `Sample_Game_3`) | Public GitHub repository (`metrica-sports/sample-data`) | Professional match analysts at Metrica Sports (`BALL_OWNING_TEAM`, `BALL_STATUS`, and event logs) | **3 full matches** (`424,108` frames @ 25 Hz) | Used for Leave-One-Game-Out cross-validation (`raw` **`macroF1 = 0.8705`**, `ProviderTouch` **`macroF1 = 0.8785 / spellF1 = 0.6935`**), 30% broadcast dropout stress tests (`aug` **`macroF1 = 0.7853`**), +45% FOV camera-pan stress tests (**`macroF1 = 0.7689`**), and `1,092` real + `1,278` synthetic 2D pass fly-by benchmarks. |
| **2. DFL Open Data (`dfl-seven-matches-v1`)** (`J03WMX`, `J03WN1`, `J03WOH`, `J03WOY`, `J03WPY`, `J03WQQ`, `J03WR9`) | Public Figshare dataset by Bassek et al. (2025) / DFL Deutsche Fußball Liga (CC-BY 4.0) | Official DFL / Sportec Solutions match annotators + TRACAB 25 Hz optical tracking | **7 full matches** (`995,280` frames @ 25 Hz) | Used in both `clean` (**`macroF1 = 0.9015`** against `ProviderTouch`, **`0.8808`** against `ProviderTeamControl`) and `missing-ball` (`17%` ball dropout, **`macroF1 = 0.8702`**) cohorts to validate cross-stadium generalization. (Also used earlier to train the discarded `MoE-HMM` baseline.) |
| **3. SkillCorner Open Data** (`1886347`, `1899585`) | Public GitHub repository (`SkillCorner/opendata`) | Automated broadcast computer vision + neural de-occlusion & possession tagging by SkillCorner | **2 full matches** (`115,347` frames @ 10 Hz) | Used in both `extrapolated` (**`macroF1 = 0.7429`**) and `broadcast-detected` (`55%` missing ball, **`macroF1 = 0.6881`**) conditions to stress-test frame-rate scaling (10 Hz vs 25 Hz) and severe broadcast camera dropouts. |
| **4. COCO Keypoints 2017 (`yolov8n-pose.pt`)** | Ultralytics / Microsoft COCO Consortium (Open Source) | COCO human keypoint crowd annotators (0 soccer match labels) | Pre-trained `6.8 MB` weights | Used without fine-tuning in `solution/referee_pose_detector.py` to extract generic human shoulder/elbow/wrist keypoints on the main referee crop. |

---

## 4. The Possession Rules We Implemented

Every threshold in `solution/touch_possession.py` maps to an interpretable physical or geometric quantity:

### A. Match Period Bounds (`X` vs. Graded Half)
- Frames inside the two official match halves (`[11185, 90652)` and `[110808, 190198)`, totaling `158,857` graded frames) are decoded into `H`, `A`, or `D`; all `40,462` pre-match, halftime, and post-match frames are labeled `X`.

### B. Restarts & Dead Balls (`D`)
1. **Out-of-Bounds Pitch Geometry**: The ball is marked `D` when its coordinates cross the official `105 m × 68 m` touchline/goal-line by more than the ball radius + homography edge tolerance (`oob_margin = 0.40 m`), **unless** all 22 players remain in high-speed live play (`med_pv > 1.38 * slowdown_med_thresh` and `p85_pv > 1.32 * slowdown_p85_thresh`), which guards against momentary camera-pan homography overshoot along the touchline.
2. **In-Bounds Stoppages (Fouls, Offsides, Goals)**: The ball transitions to `D` when collective 22-player speeds drop below the self-calibrated stoppage quantiles (`p85_pv < slowdown_p85_thresh`, calibrated at the 23rd percentile of visible play `~1.85 m/s`), when an acoustic referee whistle burst (`w_ratio > mean + 6.8 * std`) or referee gesture (`ref_signal_score`) coincides with collective player deceleration, or during post-goal dead-ball windows. When collective slowdown is confirmed, the stoppage onset rolls back up to `1.6 s` (`40` frames) to the physical whistle/foul deceleration point.
3. **Restart Confirmation & FIFA Penalty-Area Guard**: Play resumes from `D` to `H`/`A` only when a player within the touch radius (`touch_radius = 1.80 m`) imparts a physical kick (`ball_speed > 1.06 * max_dribble_speed` or `acc > 0.78 * acc_threshold`), **or** during a low-speed short goal-kick reception inside the official **FIFA Penalty Area** (`x <= 16.5 m + touch_radius` with `tang_acc < -0.35 * acc_threshold`). If a player merely nudges or places the ball during a stoppage and collective speeds remain dead (`ball_vis`, `ball_speed < 0.25 * max_dribble_speed`, `p85_pv < slowdown_p85_thresh`) within `14.4 s`, the premature restart is rolled back to `D`.

### C. Loose Balls, Non-Contact 2D Pass Fly-Bys & One-Touch Clearances
1. **Galilean-Invariant Co-Moving Dribble Control**: A player gains controlled dribble possession under two physical regimes:
   - *Standard Close Dribble*: Player is within `close_dribble_radius = 0.61 * touch_radius` (`1.10 m`) with ball speed `< max_dribble_speed` (`4.70 m/s`) for at least `min_dwell_frames = 14` (`0.56 s`).
   - *Galilean Co-Moving Fast Carry*: When a player sprints with the ball (`closest_pspeed > 0.51 * max_dribble_speed` and `ball_speed < 1.96 * max_dribble_speed ≈ 9.2 m/s`), control is maintained in the player's co-moving reference frame whenever the relative ball-player speed `||v_ball - v_player|| < 0.53 * max_dribble_speed` (`2.49 m/s`) within `0.81 * touch_radius` (`1.46 m`) for `fast_carry_dwell = 10` frames (`0.40 s`).
2. **Non-Contact 2D Pass Fly-By Rejection**: When a ball in flight (`ball_speed > 1.87 * max_dribble_speed ≈ 8.8 m/s`) passes within `1.80 m` of an opposing player in 2D projection **without** directional deflection (`dir_change < 0.14 * dir_threshold ≈ 3.4°`) or tangential deceleration (`tang_acc > -0.70 * acc_threshold`), it is classified as a **non-contact 2D fly-by** and ignored (`100.0%` rejection on `BAR-ATH`; `90.0–93.4%` real & `99.6–100.0%` synthetic rejection across `1,307` injected pass fly-overs on `Metrica`).
3. **One-Touch Power Clearances vs. Passive Brushes**: An uncontested impulsive strike (`acc > 1.15 * acc_threshold` within `0.78 * touch_radius`, or `acc > acc_threshold = 11.5 m/s²` and `dir_change > dir_threshold = 0.43 rad` within `touch_radius = 1.80 m`) transfers possession immediately. In post-processing, a brief sub-dwell spell (`< 14` frames) sandwiched between two spells of the opposing team is preserved (and its onset rolled back across the spline-smoothed deceleration reception phase) **if and only if** it exhibits a true power-clearance impulse (`max(acc) > 1.48 * acc_threshold` and exit `ball_speed > 1.70 * max_dribble_speed`); otherwise, non-intercepting pass brushes where the ball remains in flight (`min(ball_speed) > 0.34 * max_dribble_speed`) are smoothed out.

### D. Challenges, Contested Aerial Headers & Missing-Ball Bridges
1. **Contested Duel Guard**: When both teams have players within `opp_dist < closest_dist + contested_margin` (`0.42 m`) and the ball is moving rapidly (`ball_speed > 1.38 * max_dribble_speed`), single-frame touches do not flip possession until a team establishes dwell control or wins a clear uncontested impulse.
2. **Contested Aerial Header 2D Parallax Resolution**: At header height (`h ≈ 2.2 m`), 2D camera homography projects the ball `2.5–3.3 m` away from both jumping players (`min_dist_h < 3.33 m` and `min_dist_a < 3.33 m`). When the ball hits an aerial velocity apex (`ball_speed < 0.30 * max_dribble_speed`, `dir_change > 0.44 * dir_threshold`), the winning team is resolved by the post-apex longitudinal ball velocity (`+0.24 s`) confirmed by collective 22-player longitudinal acceleration (`+0.16 s`).
3. **Missing-Ball Bridges & `NaN`-Boundary Kink Suppression**: Across long camera dropouts where the ball is `NaN` or untracked (`closest_dist > 1.83 * touch_radius ≈ 3.3 m`), turnovers are localized using **collective 22-player longitudinal acceleration** (`coll_ax > coll_accel_thresh`, calibrated at `1.55 * std(coll_ax) ≈ 0.89 m/s²` in the team's attacking direction). Additionally, central-difference second derivatives (`acc`, `tang_acc`, `dir_change`) are zeroed within `±2` frames of multi-frame `NaN` linear-interpolation boundaries (`total_gap > 2`) to eliminate artificial Dirac-delta slope discontinuities.

---

## 5. Limitations & What We Would Fix With More Time

1. **Sub-Second Transition Boundary Latency & Rater Jitter (`±1–4` frames)**:
   - On the `7,500`-frame `BAR-ATH` validation window, all `67` frame errors (`0.89%` error rate) are either sub-second transition-boundary offsets (`~40` frames across 20 transitions, `median latency = -1.0 frame / 40 ms` from finite-difference derivative stencils and missing-ball interpolation) or the end-of-clip short goal-kick cutoff (`~27` frames near `18414–18685`, where the ground truth remains `D` until the 5-minute clip ends; `Spell #17` still overlaps with `IoU = 0.729 > 0.70`). We deliberately do **not** tune parameters to force these `67` boundary frames to zero: `spellF1` incurs zero penalty once every spell exceeds `IoU >= 0.70` (`spellF1 = 1.0000`), `macroF1` differences below `~0.01` are within the official noise floor (`0.9917`), and fitting `40–120 ms` operator click-timing jitter on the 5-minute slice would overfit to rater variance and hurt full-match generalization.
2. **Quick Restarts Without Collective Slowdown & Delayed Whistle Reactions**:
   - On `Metrica` and `DFL`, when a foul is called in midfield and the attacking team takes a quick free kick within `< 2 seconds`, collective 22-player speed (`med_pv`) never drops below the stoppage threshold, causing the model to miss brief `D` intervals. Conversely, during prolonged open-play injury stoppages where players jog before the ball is kicked out, stoppage onset can shift by `1–2 seconds`.
3. **Low Frame-Rate Neural-Extrapolated Tracking (`SkillCorner` 10 Hz)**:
   - On `SkillCorner` (`10 Hz`), where off-screen ball trajectories are neurally extrapolated by a smooth prior, physical kick accelerations (`acc`) are dampened, reducing `macroF1` to `0.7429` (`0.6881` under 55% missing ball).
4. **What We Would Build With More Time**:
   - **3D Ball Trajectory Reconstruction (`x, y, z, v_z`)**: Triangulating the ball's vertical height `z(t)` from the wide-angle tactical video and tight broadcast camera would directly separate high-altitude pass fly-overs (`z > 2.0 m`) from ground passes (`z ≈ 0 m`) and directly detect headers and bounces without relying on 2D ground-plane velocity and inflection proxies.
   - **End-to-End Differentiable Physics-Informed Neural ODE / Semi-Markov CRF**: Replacing hard quantile thresholds with soft differentiable transition potentials over physical impulse features while keeping strict Galilean and pitch-reflection invariance.

---

## 6. Reproduction Instructions

### Environment & Dependencies
Requires Python 3.10+ and the packages in `requirements.txt` / `requirements-solution.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-solution.txt
```

### 1. Generate and Validate the Exact Submission File (`my_submission.csv`)
With the competition data unpacked under `data/` (or `MTAH_DATA_DIR`), run:

```bash
# 1. Parse raw ATD tracking into cache/tactical_atd.npz (automatically invoked if cache is missing)
python3 solution/parse_atd.py

# 2. Run the self-calibrated physics possession model and write my_submission.csv
python3 solution/touch_possession.py --output my_submission.csv

# 3. Validate the exact submission file with the official validator
python3 evaluation/validate_submission.py my_submission.csv

# 4. Score against the public 5-minute ground truth
python3 evaluation/run_evaluation.py --gt public
```

### 2. Run Cross-Validation, 2D Fly-By Stress Tests & Multi-Dataset Benchmarks
```bash
# Run 5-fold non-public BAR-ATH CV, 3-fold Metrica LOGO-CV, and Real/Synthetic 2D Fly-By stress tests
python3 solution/cross_validate.py

# Run the full 15-match multi-dataset experiment (BAR-ATH, Metrica, DFL, SkillCorner)
python3 solution/multi_dataset_experiment.py
```

### 3. File Checksums (SHA-256)
| File | SHA-256 |
| :--- | :--- |
| `my_submission.csv` | `40801165dbd96516d6ec29b54db66cac1ed9fd35575a5e5b48a0ebb906b31caf` |
| `evaluation/results/submission_touch_v3.csv` | `40801165dbd96516d6ec29b54db66cac1ed9fd35575a5e5b48a0ebb906b31caf` |
| `solution/touch_possession.py` | `f90adc9d20c0df79cd8ad220246a771166e5a6cd87dc03b730540efbb321b007` |
| `solution/player_team_classifier.py` | `cf07dfdcb871e5c2de2f8b803cdfd41904cff88a00031d2499ac699d5165b294` |
| `solution/referee_pose_detector.py` | `2d1fd3aeda90208d8aca21d6784562ec92d012efdfd6a3bb442bdb22dcda98ac` |
| `solution/whistle_detector.py` | `9bc4edbda1763cba331fc3de0258a50f2206c80cdbd6f49debb440d2127ce0a9` |
| `solution/parse_atd.py` | `d479424a8cd6c067fdafb7a8fe3a4e4df18f1e196315eb2c2cb032381f0fda5d` |
| `solution/cross_validate.py` | `33274ff16d84f2e33b6f032bb3f9f226e31dcef2891266fe883632f38d6557c2` |
| `solution/multi_dataset_experiment.py` | `f13e81efbc4f236b008d9241675fc65b115fe2d564854d0b993a7073da3b7cf8` |
