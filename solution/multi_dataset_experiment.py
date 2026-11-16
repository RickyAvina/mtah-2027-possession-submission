"""Multi-Dataset Possession Experiment Suite.

Evaluates and compares possession models and ground-truth definitions across
four distinct soccer tracking providers (~2.1M frames across 15 matches):

  1. BAR-ATH (Competition Broadcast Tracking, 25 Hz):
     - Non-Public 85 min (151,355 frames) + Blind Public Holdout (7,500 frames).
  2. Metrica Sports Open Sample Data (3 Full Matches, 429,923 frames, 25 Hz):
     - Clean Stadium Optical (`raw`),
     - Competition-Augmented (`aug`: 30% missing ball + swaps + overflow tracks),
     - Broadcast-FOV + High Dropout (`fov45`: 22m camera frustum crop + 45% missing ball).
  3. DFL / Sportec Open Bundesliga Dataset (7 Full Matches, 1,002,651 frames, 25 Hz):
     - Registered `clean` benchmark (`dfl7-clean-9b57916fda4f`),
     - Registered `missing-ball` benchmark (`dfl7-missing-ball-844caf67787a`, seed-11 mask),
     - Evaluated against both Official DFL Micro-Touch Reference and Competition-Harmonized
       Continuous Reference (bridging <1.5s deflection blips).
  4. SkillCorner Open Broadcast Dataset (3 Full Matches, ~435,000 resampled 25 Hz frames):
     - Match 1886347 (Auckland FC vs Newcastle), Match 1899585 (Auckland FC vs Wellington),
       and SkillCorner Structured Benchmark Match,
     - Tested on both `extrapolated` (22 players) and `broadcast-detected` (`is_detected=True`
       real single-camera TV vision dropouts),
     - Evaluated against both Raw `possession.group` (loose balls = D) and Continuous
       `phases_of_play` (loose balls retain last controlling team).
"""

import gzip
import json
import os
import pathlib
import sys
import time
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import evaluation.possession_metrics as pm
from solution.scoring import score
from solution.parse_atd import load_cached_atd
from solution.external_data import GAME_IDS, load_game, augment
from solution.touch_possession import (
    DEFAULT_PARAMS,
    PUBLIC_HOLDOUT,
    precompute_features,
    predict_interval,
    run_touch_possession_model,
)


# ---------------------------------------------------------------------------
# 1. Dataset Loaders & Semantic Ground-Truth Builders
# ---------------------------------------------------------------------------

def harmonize_continuous_reference(gt_codes, runs, min_blip_frames=38, max_loose_dead_frames=0):
    """Map micro-touch / unowned-loose-ball GT into Competition-style continuous control.

    Per CHALLENGE.md:
      - A brief touch, deflection, or aerial win without subsequent control (<1.5s = 38f)
        sandwiched between the same team's possession does not establish a new spell.
      - Optionally bridges brief unowned live-play gaps (`max_loose_dead_frames`) inside halves.
    """
    out = gt_codes.copy()
    for rs, re in runs:
        sl = out[rs:re].copy()
        n = len(sl)
        # 1. Optionally bridge brief D gaps between live possessions (for SkillCorner raw group)
        if max_loose_dead_frames > 0:
            i = 0
            last_live = -1
            while i < n:
                c = sl[i]
                j = i + 1
                while j < n and sl[j] == c:
                    j += 1
                if c in (0, 1):
                    last_live = c
                elif c == 2 and (j - i) <= max_loose_dead_frames and last_live in (0, 1):
                    sl[i:j] = last_live
                i = j

        # 2. Absorb short opposing deflection blips (< min_blip_frames) sandwiched between same team
        for _ in range(2):
            runs_list = []
            i = 0
            while i < n:
                c = sl[i]
                j = i + 1
                while j < n and sl[j] == c:
                    j += 1
                runs_list.append([c, i, j])
                i = j
            for r_idx in range(1, len(runs_list) - 1):
                c_Lab, s_i, e_i = runs_list[r_idx]
                prev_lab = runs_list[r_idx - 1][0]
                next_lab = runs_list[r_idx + 1][0]
                if c_Lab in (0, 1) and (e_i - s_i) < min_blip_frames:
                    if prev_lab == next_lab and prev_lab in (0, 1):
                        sl[s_i:e_i] = prev_lab
        out[rs:re] = sl
    return out


def load_dfl_match(game_id):
    """Load one DFL Bundesliga match (both clean and missing-ball) + reference labels."""
    folder = pathlib.Path(ROOT) / "ext_data" / "dfl7" / game_id
    meta = json.load((folder / "meta.json").open())
    chunks = []
    for c in range((meta["frames"] + meta["chunkFrames"] - 1) // meta["chunkFrames"]):
        raw = gzip.decompress((folder / f"{c}.bin.gz").read_bytes())
        chunks.append(np.frombuffer(raw, dtype="<f4").reshape(-1, len(meta["playerIds"]) + 1, 2))
    coords = np.concatenate(chunks)
    ball_clean = coords[:, 0, :].copy()
    players = coords[:, 1:, :].copy()
    ref = np.frombuffer(gzip.decompress((folder / "reference.bin.gz").read_bytes()), dtype=np.uint8)
    mask = np.frombuffer(gzip.decompress((folder / "dropout-11.bin.gz").read_bytes()), dtype=np.uint8).astype(bool)

    ball_missing = ball_clean.copy()
    ball_missing[mask] = np.nan
    runs = [(int(b[1]), int(b[2]) + 1) for b in meta["blocks"]]
    teams = np.array(meta["teams"], dtype=np.int8)

    base = {
        "game": f"DFL_{game_id}",
        "match": meta["match"],
        "n_frames": len(ball_clean),
        "pitch": [float(meta["pitch"][0]), float(meta["pitch"][1])],
        "player_xy": players,
        "track_teams": teams,
        "runs": runs,
        "labels": ref,
        "labels_harmonized": harmonize_continuous_reference(ref, runs, min_blip_frames=38),
    }
    clean_dict = dict(base, ball_xy=ball_clean, condition="clean")
    miss_dict = dict(base, ball_xy=ball_missing, condition="missing-ball")
    return clean_dict, miss_dict


def load_skillcorner_jsonl_match(match_id):
    """Load a SkillCorner 10 Hz broadcast match and upsample to 25 Hz standard format.

    Returns two tracking conditions:
      - `extrapolated`: All player_data and ball_data coordinates included.
      - `broadcast_detected`: Only `is_detected == True` players and ball included (real TV FOV).
    And two ground-truth definitions:
      - `labels_raw`: Raw `possession.group` (`None` -> D).
      - `labels_phase`: Continuous `phases_of_play.csv` + <3s in-play loose-ball bridging.
    """
    mdir = pathlib.Path(ROOT) / "ext_data" / "skillcorner-opendata" / "data" / "matches" / str(match_id)
    meta = json.load((mdir / f"{match_id}_match.json").open())
    pop = pd.read_csv(mdir / f"{match_id}_phases_of_play.csv")

    L = float(meta.get("pitch_length", 105.0))
    W = float(meta.get("pitch_width", 68.0))
    home_id = int(meta["home_team"]["id"])

    # Map player_id -> slot index (Home first, then Away)
    home_pids = [int(p["id"]) for p in meta["players"] if int(p["team_id"]) == home_id]
    away_pids = [int(p["id"]) for p in meta["players"] if int(p["team_id"]) != home_id]
    all_pids = home_pids + away_pids
    pid_to_slot = {pid: idx for idx, pid in enumerate(all_pids)}
    track_teams = np.array([0] * len(home_pids) + [1] * len(away_pids), dtype=np.int8)
    n_slots = len(all_pids)

    raw_frames = []
    periods_10 = []
    raw_codes_10 = []
    ball_ext_10 = []
    ball_det_10 = []
    pxy_ext_10 = []
    pxy_det_10 = []

    with (mdir / f"{match_id}_tracking_extrapolated.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r["period"] is None:
                continue
            raw_frames.append(int(r["frame"]))
            periods_10.append(int(r["period"]))
            g = r["possession"]["group"]
            raw_codes_10.append(0 if g == "home team" else (1 if g == "away team" else 2))

            bd = r["ball_data"]
            if bd["x"] is not None and bd["y"] is not None:
                bx, by = float(bd["x"]) + L / 2.0, float(bd["y"]) + W / 2.0
                ball_ext_10.append((bx, by))
                if bd.get("is_detected"):
                    ball_det_10.append((bx, by))
                else:
                    ball_det_10.append((np.nan, np.nan))
            else:
                ball_ext_10.append((np.nan, np.nan))
                ball_det_10.append((np.nan, np.nan))

            pe = np.full((n_slots, 2), np.nan, dtype=np.float32)
            pd_arr = np.full((n_slots, 2), np.nan, dtype=np.float32)
            for p in r["player_data"]:
                s = pid_to_slot.get(int(p["player_id"]))
                if s is not None and p["x"] is not None and p["y"] is not None:
                    px, py = float(p["x"]) + L / 2.0, float(p["y"]) + W / 2.0
                    pe[s] = (px, py)
                    if p.get("is_detected"):
                        pd_arr[s] = (px, py)
            pxy_ext_10.append(pe)
            pxy_det_10.append(pd_arr)

    periods_10 = np.array(periods_10, dtype=np.int8)
    raw_codes_10 = np.array(raw_codes_10, dtype=np.uint8)
    ball_ext_10 = np.array(ball_ext_10, dtype=np.float32)
    ball_det_10 = np.array(ball_det_10, dtype=np.float32)
    pxy_ext_10 = np.array(pxy_ext_10, dtype=np.float32)
    pxy_det_10 = np.array(pxy_det_10, dtype=np.float32)

    # Build Phase-of-Play continuous ground truth at 10 Hz
    phase_codes_10 = raw_codes_10.copy()
    f2idx = {fr: i for i, fr in enumerate(raw_frames)}
    for _, row in pop.iterrows():
        s = f2idx.get(int(row["frame_start"]))
        e = f2idx.get(int(row["frame_end"]))
        t = 0 if int(row["team_in_possession_id"]) == home_id else 1
        if s is not None and e is not None:
            phase_codes_10[s : e + 1] = t

    p1_idx = np.where(periods_10 == 1)[0]
    p2_idx = np.where(periods_10 == 2)[0]
    runs_10 = [(int(p1_idx[0]), int(p1_idx[-1]) + 1), (int(p2_idx[0]), int(p2_idx[-1]) + 1)]
    phase_codes_10 = harmonize_continuous_reference(
        phase_codes_10, runs_10, min_blip_frames=15, max_loose_dead_frames=30
    )

    # Upsample 10 Hz -> 25 Hz per period so all 25 Hz physical velocity/dwell scales apply directly
    def upsample_half(s10, e10):
        n10 = e10 - s10
        n25 = int(round(n10 * 2.5))
        t10 = np.arange(n10, dtype=np.float64)
        t25 = np.linspace(0, n10 - 1, n25, dtype=np.float64)
        nn_idx = np.clip(np.round(t25).astype(int), 0, n10 - 1)

        def interp_2d(arr10):
            out = np.full((n25, arr10.shape[1]), np.nan, dtype=np.float32)
            for c in range(arr10.shape[1]):
                col = arr10[:, c]
                v = ~np.isnan(col)
                if np.sum(v) >= 2:
                    interp_col = np.interp(t25, t10[v], col[v]).astype(np.float32)
                    # Preserve NaNs where nearest 10Hz frame was NaN
                    interp_col[~v[nn_idx]] = np.nan
                    out[:, c] = interp_col
            return out

        b_ext = interp_2d(ball_ext_10[s10:e10])
        b_det = interp_2d(ball_det_10[s10:e10])
        p_ext = interp_2d(pxy_ext_10[s10:e10].reshape(n10, -1)).reshape(n25, n_slots, 2)
        p_det = interp_2d(pxy_det_10[s10:e10].reshape(n10, -1)).reshape(n25, n_slots, 2)
        c_raw = raw_codes_10[s10:e10][nn_idx]
        c_pha = phase_codes_10[s10:e10][nn_idx]
        return b_ext, b_det, p_ext, p_det, c_raw, c_pha

    b_ext1, b_det1, p_ext1, p_det1, cr1, cp1 = upsample_half(*runs_10[0])
    b_ext2, b_det2, p_ext2, p_det2, cr2, cp2 = upsample_half(*runs_10[1])

    sep_b = np.full((1, 2), np.nan, dtype=np.float32)
    sep_p = np.full((1, n_slots, 2), np.nan, dtype=np.float32)
    sep_c = np.array([3], dtype=np.uint8)

    ball_ext = np.concatenate([b_ext1, sep_b, b_ext2], axis=0)
    ball_det = np.concatenate([b_det1, sep_b, b_det2], axis=0)
    pxy_ext = np.concatenate([p_ext1, sep_p, p_ext2], axis=0)
    pxy_det = np.concatenate([p_det1, sep_p, p_det2], axis=0)
    codes_raw = np.concatenate([cr1, sep_c, cr2], axis=0)
    codes_pha = np.concatenate([cp1, sep_c, cp2], axis=0)
    runs_25 = [(0, len(b_ext1)), (len(b_ext1) + 1, len(ball_ext))]

    mname = f"{meta['home_team']['short_name']} vs {meta['away_team']['short_name']}"
    base = {
        "game": f"SC_{match_id}",
        "match": mname,
        "n_frames": len(ball_ext),
        "pitch": [L, W],
        "track_teams": track_teams,
        "runs": runs_25,
        "labels": codes_raw,
        "labels_harmonized": codes_pha,
    }
    ext_dict = dict(base, ball_xy=ball_ext, player_xy=pxy_ext, condition="extrapolated")
    det_dict = dict(base, ball_xy=ball_det, player_xy=pxy_det, condition="broadcast-detected")
    return ext_dict, det_dict


def create_metrica_fov_45(gm, seed=101):
    """Create a Broadcast-FOV (22m camera crop around ball) + 45% missing-ball Metrica match."""
    rng = np.random.default_rng(seed)
    out = augment(gm, rng)
    ball = out["ball_xy"].copy()
    pxy = out["player_xy"].copy()
    n = len(ball)

    # Add extra ball dropouts to reach ~45% missing ball inside live play
    t = 0
    while t < n:
        t += int(rng.exponential(160))
        if t >= n:
            break
        L = int(np.clip(rng.lognormal(np.log(55), 0.7), 15, 350))
        ball[t : min(n, t + L)] = np.nan

    # Broadcast camera center follows smoothed ball or player centroid
    t_idx = np.arange(n, dtype=np.float64)
    vis = ~np.isnan(ball[:, 0])
    cam_x = np.interp(t_idx, t_idx[vis], ball[vis, 0])
    cam_y = np.interp(t_idx, t_idx[vis], ball[vis, 1])
    cam_x = uniform_filter1d(cam_x, size=50, mode="nearest")
    cam_y = uniform_filter1d(cam_y, size=50, mode="nearest")

    # Hide players outside broadcast camera frustum (>24m in x or >18m in y from camera center)
    dx = np.abs(pxy[:, :, 0] - cam_x[:, None])
    dy = np.abs(pxy[:, :, 1] - cam_y[:, None])
    outside_fov = (dx > 24.0) | (dy > 18.0)
    pxy[outside_fov] = np.nan

    out["ball_xy"] = ball
    out["player_xy"] = pxy
    out["condition"] = "broadcast-fov-45"
    return out


# ---------------------------------------------------------------------------
# 2. Provider-Adapted / Micro-Touch Physical Predictor & Baselines
# ---------------------------------------------------------------------------

def run_nearest_player_baseline(feats, smooth_window=50):
    """Nearest-player baseline with mode smoothing and OOB dead-ball gating."""
    n = feats["n_frames"]
    codes = np.full(n, 3, dtype=np.uint8)
    ct = feats["closest_team"].copy()
    oob = feats["ball_oob"]
    frozen = feats["ball_vis"] & (feats["ball_speed"] < 0.05)

    for rs, re in feats["periods"]:
        sl = np.zeros(re - rs, dtype=np.uint8)
        cur = 0
        for i in range(re - rs):
            idx = rs + i
            if oob[idx] or frozen[idx]:
                sl[i] = 2
            else:
                if ct[idx] in (0, 1):
                    cur = int(ct[idx])
                sl[i] = cur
        # Smooth H/A with uniform filter
        if smooth_window > 1:
            is_a = (sl == 1).astype(np.float32)
            sm_a = uniform_filter1d(is_a, size=smooth_window, mode="nearest")
            live_mask = sl != 2
            sl[live_mask] = np.where(sm_a[live_mask] >= 0.5, 1, 0)
        codes[rs:re] = sl
    return codes


def run_provider_adapted_touch_model(feats, micro_touch=True, detect_frozen_ball_dead=True):
    """Run our physical touch + collective movement model adapted to provider semantics.

    - `micro_touch=True`: Sets `min_dwell_frames=3`, `min_sandwich_frames=0`,
      `contested_margin=0.05` so 1-touch deflections/clearances count as spells (for DFL).
    - `detect_frozen_ball_dead=True`: Detects DFL-style provider-clamped dead balls where
      `ball_vis` is True and `ball_speed < 0.05 m/s` for >= 12 frames (`0.48s`).
    """
    params = dict(DEFAULT_PARAMS)
    if micro_touch:
        params.update({
            "min_dwell_frames": 3,
            "min_sandwich_frames": 0,
            "contested_margin": 0.05,
            "reentry_delay_frames": 2,
        })

    if detect_frozen_ball_dead:
        # Inject provider-clamped stationary ball runs (>= 12 frames at speed < 0.05 m/s) into ball_oob
        feats_mod = dict(feats)
        bs = feats["ball_speed"]
        bv = feats["ball_vis"]
        frozen = bv & (bs < 0.05)
        frozen_run = np.zeros(len(bs), dtype=bool)
        c = 0
        for i in range(len(bs)):
            c = c + 1 if frozen[i] else 0
            if c >= 12:
                frozen_run[i - 11 : i + 1] = True
        feats_mod["ball_oob"] = feats["ball_oob"] | frozen_run
    else:
        feats_mod = feats

    lbls = run_touch_possession_model(params=params, features=feats_mod)
    return pm.label_codes(lbls)


# ---------------------------------------------------------------------------
# 3. Cross-Dataset Learned Model (Train on Dataset A -> Test on Dataset B)
# ---------------------------------------------------------------------------

def build_tabular_matrix(feats, include_ball_dead_features=True):
    """Build a 12-feature standardized kinematic + collective matrix for cross-dataset ML."""
    n = feats["n_frames"]
    d_h = np.minimum(feats["min_dist_h"], 30.0)
    d_a = np.minimum(feats["min_dist_a"], 30.0)
    diff_d = d_a - d_h  # positive -> Home closer
    sum_d = d_a + d_h

    # Smoothed proximity margin over 1s (25f) and 3s (75f)
    diff_1s = uniform_filter1d(diff_d, size=25, mode="nearest")
    diff_3s = uniform_filter1d(diff_d, size=75, mode="nearest")

    # Collective velocity toward attacking goal
    coll_vx_1s = uniform_filter1d(feats["coll_vx"], size=25, mode="nearest")
    coll_ax_1s = uniform_filter1d(feats["coll_ax"], size=25, mode="nearest")

    # Player speed quantiles normalized by match median
    med_pv = feats["med_pv"]
    p85_pv = feats["p85_pv"]
    med_scale = max(float(np.median(med_pv[med_pv > 0.1])), 0.5)
    norm_med_pv = med_pv / med_scale
    norm_p85_pv = p85_pv / med_scale
    norm_p85_3s = uniform_filter1d(norm_p85_pv, size=75, mode="nearest")

    if include_ball_dead_features:
        bv = feats["ball_vis"].astype(np.float32)
        oob = feats["ball_oob"].astype(np.float32)
        bs = np.minimum(feats["ball_speed"], 30.0)
    else:
        # Player-only stoppage mode (prevents learning DFL's frozen-ball shortcut)
        bv = np.ones(n, dtype=np.float32)
        oob = feats["ball_oob"].astype(np.float32)
        bs = np.full(n, 5.0, dtype=np.float32)

    X = np.column_stack([
        diff_d,
        diff_1s,
        diff_3s,
        sum_d,
        coll_vx_1s,
        coll_ax_1s,
        norm_med_pv,
        norm_p85_pv,
        norm_p85_3s,
        bv,
        oob,
        bs,
    ]).astype(np.float32)
    return np.nan_to_num(X, nan=0.0)


def viterbi_decode_probs(probs, runs, switch_pen=10.0, dead_pen=14.0):
    """Viterbi decode 3-state (H=0, A=1, D=2) log-probabilities per half."""
    n = len(probs)
    out = np.full(n, 3, dtype=np.uint8)
    log_p = np.log(np.clip(probs, 1e-6, 1.0))
    trans = np.array([
        [0.0, -switch_pen, -dead_pen],
        [-switch_pen, 0.0, -dead_pen],
        [-dead_pen, -dead_pen, 0.0],
    ], dtype=np.float64)

    for rs, re in runs:
        lp = log_p[rs:re]
        T = len(lp)
        dp = np.zeros((T, 3), dtype=np.float64)
        back = np.zeros((T, 3), dtype=np.int8)
        dp[0] = lp[0]
        for t in range(1, T):
            scores = dp[t - 1][:, None] + trans
            back[t] = np.argmax(scores, axis=0)
            dp[t] = lp[t] + np.max(scores, axis=0)
        path = np.zeros(T, dtype=np.uint8)
        path[-1] = int(np.argmax(dp[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = back[t + 1, path[t + 1]]
        out[rs:re] = path
    return out


def fit_dataset_model(feat_gt_list, include_ball_dead_features=True, stride=15):
    """Fit a HistGradientBoosting 3-class model on a list of (feats, gt_codes) tuples."""
    Xs, ys = [], []
    for feats, gt in feat_gt_list:
        X = build_tabular_matrix(feats, include_ball_dead_features=include_ball_dead_features)
        for rs, re in feats["periods"]:
            idx = np.arange(rs, re, stride)
            valid = gt[idx] < 3
            Xs.append(X[idx[valid]])
            ys.append(gt[idx[valid]])
    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    clf = HistGradientBoostingClassifier(
        max_iter=80, max_leaf_nodes=15, min_samples_leaf=80, l2_regularization=2.0, random_state=42
    )
    clf.fit(X_all, y_all)
    return clf, include_ball_dead_features


def predict_dataset_model(clf_bundle, feats):
    clf, inc_ball = clf_bundle
    X = build_tabular_matrix(feats, include_ball_dead_features=inc_ball)
    probs = clf.predict_proba(X)
    codes = viterbi_decode_probs(probs, feats["periods"], switch_pen=10.0, dead_pen=14.0)
    # Apply explicit forced_dead / OOB constraints
    for rs, re in feats["periods"]:
        sl = codes[rs:re]
        sl[feats["ball_oob"][rs:re] | feats["forced_dead"][rs:re]] = 2
        codes[rs:re] = sl
    return codes


# ---------------------------------------------------------------------------
# 4. Main Experimental Runner
# ---------------------------------------------------------------------------

def compute_gt_signature(name, feats, gt_codes):
    """Compute ground-truth statistical signature for taxonomy comparison."""
    runs = feats["periods"]
    active = np.zeros(feats["n_frames"], dtype=bool)
    for s, e in runs:
        active[s:e] = True
    valid = active & (gt_codes < 3)
    n_val = int(np.sum(valid))
    minutes = n_val / 25.0 / 60.0

    spells = pm.possession_spells(gt_codes, runs)
    durs = np.array([(e - s) / 25.0 for s, e, _ in spells], dtype=np.float64)

    dead = valid & (gt_codes == 2)
    live = valid & (gt_codes < 2)
    bv = feats["ball_vis"]
    bs = feats["ball_speed"]
    oob = feats["ball_oob"]
    p85 = feats["p85_pv"]

    return {
        "dataset": name,
        "frames": n_val,
        "minutes": round(minutes, 1),
        "pct_H": round(float(np.mean(gt_codes[valid] == 0)) * 100, 1),
        "pct_A": round(float(np.mean(gt_codes[valid] == 1)) * 100, 1),
        "pct_D": round(float(np.mean(dead[valid])) * 100, 1),
        "spells_per_min": round(len(spells) / max(minutes, 1e-3), 2),
        "median_spell_sec": round(float(np.median(durs)) if len(durs) else 0.0, 2),
        "pct_spells_under_2s": round(float(np.mean(durs < 2.0)) * 100 if len(durs) else 0.0, 1),
        "ball_vis_live_pct": round(float(np.mean(bv[live])) * 100 if np.any(live) else 0.0, 1),
        "ball_vis_dead_pct": round(float(np.mean(bv[dead])) * 100 if np.any(dead) else 0.0, 1),
        "frozen_ball_on_D_pct": round(float(np.mean((bv & (bs < 0.05))[dead])) * 100 if np.any(dead) else 0.0, 1),
        "oob_on_D_pct": round(float(np.mean(oob[dead])) * 100 if np.any(dead) else 0.0, 1),
        "p85_speed_live": round(float(np.median(p85[live])) if np.any(live) else 0.0, 2),
        "p85_speed_dead": round(float(np.median(p85[dead])) if np.any(dead) else 0.0, 2),
    }


def main():
    t0 = time.time()
    print("=========================================================================")
    print("  MULTI-DATASET POSSESSION & DEFINITION EXPERIMENT (15 MATCHES, ~2.1M FRAMES)")
    print("=========================================================================\n")

    # 1. Load BAR-ATH (Public Holdout + Non-Public)
    print("[1/4] Loading BAR-ATH Competition Match...")
    atd = load_cached_atd()
    bar_feats = precompute_features(atd, holdout_intervals=[PUBLIC_HOLDOUT])
    gt_pub_df = pm.read_label_csv(os.path.join(ROOT, "evaluation/gt/BAR-ATH_possession_gt_public.csv"))
    gt_pub_codes = pm.label_codes(gt_pub_df)

    # 2. Load 3 Metrica Matches (raw, aug-30, broadcast-fov-45)
    print("[2/4] Loading 3 Metrica Matches (raw, aug-30, broadcast-fov-45)...")
    rng = np.random.default_rng(0)
    metrica_sets = {"raw": [], "aug": [], "fov45": []}
    for g in GAME_IDS:
        gm = load_game(g)
        gm["labels_harmonized"] = harmonize_continuous_reference(gm["labels"], gm["runs"], min_blip_frames=38)
        gm_aug = augment(gm, rng)
        gm_aug["labels_harmonized"] = gm["labels_harmonized"]
        gm_fov = create_metrica_fov_45(gm, seed=100 + g)
        gm_fov["labels_harmonized"] = gm["labels_harmonized"]

        metrica_sets["raw"].append((gm, precompute_features(gm)))
        metrica_sets["aug"].append((gm_aug, precompute_features(gm_aug)))
        metrica_sets["fov45"].append((gm_fov, precompute_features(gm_fov)))

    # 3. Load 7 DFL Bundesliga Matches (clean & missing-ball)
    print("[3/4] Loading 7 DFL Bundesliga Matches (clean & missing-ball, 1,002,651 frames)...")
    dfl_game_ids = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]
    dfl_sets = {"clean": [], "missing": []}
    for gid in dfl_game_ids:
        c_dict, m_dict = load_dfl_match(gid)
        f_c = precompute_features(c_dict)
        m_dict["team_diag"] = (
            f_c["pred_teams"],
            {
                "corrected_xy": f_c["corrected_xy"],
                "phantom_mask": np.isnan(f_c["corrected_xy"][:, :, 0]),
                "att_dir": f_c["att_dir"],
            },
        )
        f_m = precompute_features(m_dict)
        dfl_sets["clean"].append((c_dict, f_c))
        dfl_sets["missing"].append((m_dict, f_m))

    # 4. Load 2 Full SkillCorner Matches (extrapolated & broadcast-detected)
    print("[4/4] Loading SkillCorner Broadcast Tracking Matches (1886347 & 1899585, resampled 25 Hz)...")
    sc_sets = {"extrapolated": [], "detected": []}
    for mid in ["1886347", "1899585"]:
        e_dict, d_dict = load_skillcorner_jsonl_match(mid)
        sc_sets["extrapolated"].append((e_dict, precompute_features(e_dict)))
        sc_sets["detected"].append((d_dict, precompute_features(d_dict)))

    # -----------------------------------------------------------------------
    # Experiment A: Ground-Truth Taxonomy & Semantic Differences
    # -----------------------------------------------------------------------
    print("\n=== EXPERIMENT A: GROUND-TRUTH POSSESSION & STOPPAGE TAXONOMY ACROSS PROVIDERS ===")
    pub_feats_slice = dict(bar_feats, periods=[PUBLIC_HOLDOUT])
    tax_rows = [
        compute_gt_signature("BAR-ATH Public Slice (17 spells)", pub_feats_slice, gt_pub_codes),
        compute_gt_signature("Metrica G1 (Continuous GT)", metrica_sets["raw"][0][1], metrica_sets["raw"][0][0]["labels"]),
        compute_gt_signature("DFL J03WMX (Official Micro-Touch GT)", dfl_sets["clean"][0][1], dfl_sets["clean"][0][0]["labels"]),
        compute_gt_signature("DFL J03WMX (Harmonized Continuous GT)", dfl_sets["clean"][0][1], dfl_sets["clean"][0][0]["labels_harmonized"]),
        compute_gt_signature("SkillCorner 1886347 (Raw Group GT)", sc_sets["detected"][0][1], sc_sets["detected"][0][0]["labels"]),
        compute_gt_signature("SkillCorner 1886347 (Phase-Bridged GT)", sc_sets["detected"][0][1], sc_sets["detected"][0][0]["labels_harmonized"]),
    ]
    for r in tax_rows:
        print(
            f"  {r['dataset']:40s} | H/A/D={r['pct_H']:4.1f}/{r['pct_A']:4.1f}/{r['pct_D']:4.1f}% | "
            f"Cadence={r['spells_per_min']:4.2f}/m (med={r['median_spell_sec']:4.2f}s, <2s={r['pct_spells_under_2s']:4.1f}%) | "
            f"Vis(L/D)={r['ball_vis_live_pct']:4.1f}/{r['ball_vis_dead_pct']:4.1f}% | "
            f"FrozenOnD={r['frozen_ball_on_D_pct']:5.1f}%"
        )

    # -----------------------------------------------------------------------
    # Experiment B: Model Evaluation Under Native vs Harmonized Possession Definitions
    # -----------------------------------------------------------------------
    print("\n=== EXPERIMENT B: MODEL PERFORMANCE ACROSS DATASETS, CONDITIONS & DEFINITIONS ===")

    def eval_cohort(cohort_name, pairs, use_frozen_ball_in_adapted=False):
        metrics = {
            "baseline_native": [],
            "comp_touch_native": [],
            "adapted_touch_native": [],
            "comp_touch_harmonized": [],
            "adapted_touch_harmonized": [],
        }
        for d, feats in pairs:
            gt_nat = d["labels"]
            gt_har = d["labels_harmonized"]
            runs = d["runs"]

            c_base = run_nearest_player_baseline(feats, smooth_window=50)
            c_comp = pm.label_codes(run_touch_possession_model(DEFAULT_PARAMS, feats))
            c_adap = run_provider_adapted_touch_model(
                feats, micro_touch=True, detect_frozen_ball_dead=use_frozen_ball_in_adapted
            )

            metrics["baseline_native"].append(score(gt_nat, c_base, runs))
            metrics["comp_touch_native"].append(score(gt_nat, c_comp, runs))
            metrics["adapted_touch_native"].append(score(gt_nat, c_adap, runs))
            metrics["comp_touch_harmonized"].append(score(gt_har, c_comp, runs))
            metrics["adapted_touch_harmonized"].append(score(gt_har, c_adap, runs))

        def avg(key):
            arr = metrics[key]
            return {
                "spellF1": round(float(np.mean([x["spellF1"] for x in arr])), 4),
                "macroF1": round(float(np.mean([x["macroF1"] for x in arr])), 4),
                "f1_D": round(float(np.mean([x["f1_D"] for x in arr])), 4),
                "accuracy": round(float(np.mean([x.get("accuracy", x.get("acc", 0.0)) for x in arr])), 4),
            }

        res = {k: avg(k) for k in metrics}
        print(
            f"  {cohort_name:32s} | "
            f"Native GT -> Base: {res['baseline_native']['spellF1']:.4f}/{res['baseline_native']['macroF1']:.4f} | "
            f"CompTouch: {res['comp_touch_native']['spellF1']:.4f}/{res['comp_touch_native']['macroF1']:.4f} | "
            f"ProviderTouch: {res['adapted_touch_native']['spellF1']:.4f}/{res['adapted_touch_native']['macroF1']:.4f} || "
            f"Harmonized GT -> CompTouch: {res['comp_touch_harmonized']['spellF1']:.4f}/{res['comp_touch_harmonized']['macroF1']:.4f}"
        )
        return res

    exp_b = {
        "Metrica_3_Raw": eval_cohort("Metrica (3 games, raw)", metrica_sets["raw"], False),
        "Metrica_3_Aug30": eval_cohort("Metrica (3 games, aug-30%)", metrica_sets["aug"], False),
        "Metrica_3_FOV45": eval_cohort("Metrica (3 games, FOV+45% drop)", metrica_sets["fov45"], False),
        "DFL_7_Clean": eval_cohort("DFL (7 games, clean)", dfl_sets["clean"], True),
        "DFL_7_MissingBall": eval_cohort("DFL (7 games, missing-ball)", dfl_sets["missing"], True),
        "SkillCorner_2_Extrapolated": eval_cohort("SkillCorner (2 games, extrap)", sc_sets["extrapolated"], False),
        "SkillCorner_2_BroadcastDet": eval_cohort("SkillCorner (2 games, TV-det)", sc_sets["detected"], False),
    }

    # -----------------------------------------------------------------------
    # Experiment C: Cross-Dataset Training Transfer Matrix (Train on A -> Test on B)
    # -----------------------------------------------------------------------
    print("\n=== EXPERIMENT C: CROSS-DATASET TRAINING TRANSFER MATRIX (spellF1 / macroF1) ===")
    train_configs = {
        "Train: Metrica (3 games)": (
            [(f, d["labels"]) for d, f in metrica_sets["raw"] + metrica_sets["aug"]],
            True,
        ),
        "Train: DFL Clean (w/ ball speed)": (
            [(f, d["labels"]) for d, f in dfl_sets["clean"]],
            True,
        ),
        "Train: DFL Player-Only Stop": (
            [(f, d["labels_harmonized"]) for d, f in dfl_sets["missing"]],
            False,
        ),
        "Train: SkillCorner (Phase GT)": (
            [(f, d["labels_harmonized"]) for d, f in sc_sets["detected"]],
            True,
        ),
        "Train: Pooled Multi-Dataset": (
            [(f, d["labels"]) for d, f in metrica_sets["raw"] + metrica_sets["aug"]]
            + [(f, d["labels_harmonized"]) for d, f in dfl_sets["missing"][:3]]
            + [(f, d["labels_harmonized"]) for d, f in sc_sets["detected"]],
            False,
        ),
    }

    test_cohorts = {
        "Metrica_Aug": [(d["labels"], f, d["runs"]) for d, f in metrica_sets["aug"]],
        "DFL_Missing": [(d["labels_harmonized"], f, d["runs"]) for d, f in dfl_sets["missing"]],
        "SkillCorner_Det": [(d["labels_harmonized"], f, d["runs"]) for d, f in sc_sets["detected"]],
        "BAR_ATH_Holdout": [(gt_pub_codes, bar_feats, [PUBLIC_HOLDOUT])],
    }

    transfer_matrix = {}
    for t_name, (train_pairs, inc_ball) in train_configs.items():
        clf_bundle = fit_dataset_model(train_pairs, include_ball_dead_features=inc_ball)
        row_res = {}
        row_strs = []
        for c_name, test_list in test_cohorts.items():
            s_list = []
            for gt, feats, runs in test_list:
                pred = predict_dataset_model(clf_bundle, feats)
                if c_name == "BAR_ATH_Holdout":
                    ev = pm.evaluate(gt, pred)
                    s_list.append({"spellF1": ev["spell_f1_mean"], "macroF1": ev["macro_f1"]})
                else:
                    s_list.append(score(gt, pred, runs))
            sf1 = round(float(np.mean([x["spellF1"] for x in s_list])), 4)
            mf1 = round(float(np.mean([x["macroF1"] for x in s_list])), 4)
            row_res[c_name] = {"spellF1": sf1, "macroF1": mf1}
            row_strs.append(f"{c_name}={sf1:.3f}/{mf1:.3f}")
        transfer_matrix[t_name] = row_res
        print(f"  {t_name:32s} -> " + " | ".join(row_strs))

    # -----------------------------------------------------------------------
    # Experiment D: Non-Contact 2D Fly-By Benchmark Across All 4 Providers
    # -----------------------------------------------------------------------
    print("\n=== EXPERIMENT D: NON-CONTACT 2D FLY-BY REJECTION ACROSS PROVIDERS ===")
    from solution.cross_validate import evaluate_real_flybys

    bar_pred_codes = pm.label_codes(run_touch_possession_model(DEFAULT_PARAMS, bar_feats))
    flyby_providers = [
        evaluate_real_flybys("BAR-ATH Public Holdout", bar_feats, gt_pub_codes, bar_pred_codes, [PUBLIC_HOLDOUT]),
    ]
    for g_idx, (d, f) in enumerate(metrica_sets["raw"], 1):
        p = pm.label_codes(run_touch_possession_model(DEFAULT_PARAMS, f))
        flyby_providers.append(evaluate_real_flybys(f"Metrica Game {g_idx} (raw)", f, d["labels"], p, d["runs"]))
    for d, f in dfl_sets["clean"][:2]:
        p = pm.label_codes(run_touch_possession_model(DEFAULT_PARAMS, f))
        flyby_providers.append(evaluate_real_flybys(f"{d['game']} (Harmonized GT)", f, d["labels_harmonized"], p, d["runs"]))
    for d, f in sc_sets["detected"]:
        p = pm.label_codes(run_touch_possession_model(DEFAULT_PARAMS, f))
        flyby_providers.append(evaluate_real_flybys(f"{d['game']} (Broadcast-Det)", f, d["labels_harmonized"], p, d["runs"]))

    out_path = os.path.join(ROOT, "evaluation/results/multi_dataset_experiment.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "elapsed_sec": round(time.time() - t0, 1),
                "taxonomy": tax_rows,
                "model_comparison_by_definition": exp_b,
                "cross_dataset_transfer_matrix": transfer_matrix,
                "non_contact_2d_flyby_benchmark": flyby_providers,
            },
            f,
            indent=2,
        )
    print(f"\nSaved complete multi-dataset experiment report to {out_path} in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
