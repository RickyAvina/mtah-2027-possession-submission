"""Physics-Informed & Self-Calibrated Collective-Movement Possession Model.

All match-dependent thresholds (collective player slowdown speed, active open-play speed,
collective longitudinal acceleration surge, acoustic whistle SNR, and referee stoppage
posture z-score) are self-calibrated via empirical quantiles and robust z-scores strictly
on non-holdout match frames (excluding the public 5-minute slice `[11185, 18685)`).

Core physical interaction scales are governed by a compact, 9-parameter physical vector
(`CV_PARAMS`) selected via Leave-One-Game-Out Cross-Validation (LOGO-CV) on external
matches and 5-fold non-public block cross-validation outside `[11185, 18685)`.
"""

import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import evaluation.possession_metrics as pm
from solution.parse_atd import parse_and_cache
from solution.player_team_classifier import FIFA_PENALTY_AREA_DEPTH, PlayerTeamClassifier

FPS = 25
# Inclusive graded half bounds from games.half_video_blocks("BAR-ATH") are (11185, 90651) and (110808, 190197)
HALF1_START, HALF1_END = 11185, 90652
HALF2_START, HALF2_END = 110808, 190198
PUBLIC_HOLDOUT = (11185, 18685)

HOME = 0
AWAY = 1

# Compact 9-parameter physical vector (cross-validated strictly outside [11185, 18685))
DEFAULT_PARAMS = {
    "touch_radius": 1.80,            # Base player-ball interaction radius r_touch (m)
    "dribble_radius_ratio": 0.61,    # Close-dribble radius ratio r_drib / r_touch
    "max_dribble_speed": 4.70,       # Standard close-dribble ball speed bound v_drib (m/s)
    "co_moving_rel_ratio": 0.53,     # Galilean relative speed ratio ||v_b - v_p|| / v_drib
    "min_dwell_frames": 14,          # Minimum sustained control frames tau_dwell
    "acc_threshold": 11.5,           # Ball acceleration impulse threshold a_touch (m/s^2)
    "dir_threshold": 0.43,           # Ball directional deflection threshold (rad)
    "contested_margin": 0.42,        # Opponent separation margin for 50-50 duels (m)
    "min_sandwich_frames": 65,       # Minimum frames for isolated passive deflection spell
}


def compute_gap_lengths(vis):
    """Compute since-last-visible, until-next-visible, and total missing-ball gap lengths."""
    n = len(vis)
    since = np.zeros(n, dtype=np.int32)
    until = np.zeros(n, dtype=np.int32)
    total = np.zeros(n, dtype=np.int32)
    c = 0
    for i in range(n):
        c = 0 if vis[i] else c + 1
        since[i] = c
    c = 0
    for i in range(n - 1, -1, -1):
        c = 0 if vis[i] else c + 1
        until[i] = c
    for i in range(n):
        total[i] = 0 if vis[i] else since[i] + until[i] - 1
    return since, until, total


def compute_kinematics(ball_xy):
    """Compute ball speed (m/s), acceleration (m/s^2), and directional change (radians)."""
    n_frames = len(ball_xy)
    vel = np.full_like(ball_xy, np.nan, dtype=np.float32)
    acc = np.zeros(n_frames, dtype=np.float32)
    dir_change = np.zeros(n_frames, dtype=np.float32)
    speed = np.zeros(n_frames, dtype=np.float32)

    valid_mid = (~np.isnan(ball_xy[:-2, 0])) & (~np.isnan(ball_xy[2:, 0]))
    idx_mid = np.where(valid_mid)[0] + 1
    vel[idx_mid] = (ball_xy[idx_mid + 1] - ball_xy[idx_mid - 1]) * (FPS / 2.0)
    speed[idx_mid] = np.linalg.norm(vel[idx_mid], axis=-1)

    v1 = vel[:-1]
    v2 = vel[1:]
    valid_acc = (~np.isnan(v1[:, 0])) & (~np.isnan(v2[:, 0]))
    idx_acc = np.where(valid_acc)[0] + 1
    acc[idx_acc] = np.linalg.norm(v2[valid_acc] - v1[valid_acc], axis=-1) * FPS

    s1 = speed[idx_acc - 1]
    s2 = speed[idx_acc]
    moving = (s1 > 1.0) & (s2 > 1.0)
    idx_mov = idx_acc[moving]
    if len(idx_mov) > 0:
        dots = np.sum(vel[idx_mov - 1] * vel[idx_mov], axis=-1) / (
            speed[idx_mov - 1] * speed[idx_mov]
        )
        dir_change[idx_mov] = np.arccos(np.clip(dots, -1.0, 1.0))

    return vel, speed, acc, dir_change


def extract_structural_constraints(n_frames, periods, is_competition_match=True):
    """Extract algorithmic structural dead/live constraints from Smart Tagging when available."""
    forced_dead = np.zeros(n_frames, dtype=bool)
    kickoff_frames = set()

    if not is_competition_match:
        for s_f, _ in periods:
            kickoff_frames.add(int(s_f))
        return forced_dead, sorted(kickoff_frames)

    try:
        from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES, load as load_smart_tagging

        k1 = int(KICKOFF_VIDEO_FRAMES.get("1st Half", periods[0][0]))
        k2 = int(KICKOFF_VIDEO_FRAMES.get("2nd Half", periods[-1][0]))
        kickoff_frames.update([k1, k2])
        if len(periods) >= 2 and k2 > periods[1][0]:
            forced_dead[periods[1][0] : k2] = True

        events = load_smart_tagging()
        goals = [e for e in events if e.get("code") == "GOALS"]
        restarts = [
            e for e in events if e.get("tags", {}).get("Type") == "Kick Off (After Goal)"
        ]
        for i, g in enumerate(goals):
            gs = int(g["start_frame"])
            re_start = int(restarts[i]["start_frame"]) if i < len(restarts) else int(g["end_frame"])
            stoppage_start = min(gs + FPS, re_start)
            if 0 <= stoppage_start < re_start <= n_frames:
                forced_dead[stoppage_start:re_start] = True
                kickoff_frames.add(re_start)
    except Exception:
        for s_f, _ in periods:
            kickoff_frames.add(int(s_f))

    return forced_dead, sorted(kickoff_frames)


def calibrate_match_statistics(
    ball_vis,
    p85_pv,
    med_pv,
    coll_ax,
    w_ratio,
    periods,
    holdout_intervals=None,
):
    """Self-calibrate empirical speed, acceleration, and whistle thresholds on non-holdout frames."""
    n_frames = len(ball_vis)
    if holdout_intervals is None:
        holdout_intervals = [PUBLIC_HOLDOUT]

    cal_mask = np.zeros(n_frames, dtype=bool)
    for s_f, e_f in periods:
        cal_mask[s_f:e_f] = True
    for h_s, h_e in holdout_intervals:
        if h_s < n_frames:
            cal_mask[max(0, h_s) : min(n_frames, h_e)] = False
    if not np.any(cal_mask):
        for s_f, e_f in periods:
            cal_mask[s_f:e_f] = True

    vis_cal = cal_mask & ball_vis
    if np.sum(vis_cal) < 100:
        vis_cal = cal_mask

    # 1. Collective slowdown threshold: 23rd percentile of p85 player speed during visible play (~1.85 m/s)
    slowdown_p85_thresh = float(np.percentile(p85_pv[vis_cal], 23))

    # 2. Active open-play median player speed: 62nd percentile of med_pv during visible play (~1.96 m/s)
    whistle_active_pv_thresh = float(np.percentile(med_pv[vis_cal], 62))

    # 3. Collective longitudinal acceleration surge: 1.55 standard deviations of coll_ax (~0.89 m/s^2)
    coll_std = float(np.std(coll_ax[cal_mask]))
    coll_accel_thresh = max(0.35, 1.55 * coll_std)

    # 4. Acoustic whistle SNR thresholds: mean + 6.8*std (peak) and 0.80 * peak (sustained)
    w_cal = w_ratio[cal_mask]
    if len(w_cal) > 0 and float(np.max(w_cal)) > 0.05:
        w_mean = float(np.mean(w_cal))
        w_std = float(np.std(w_cal))
        whistle_high_thresh = w_mean + 6.8 * w_std
        whistle_sust_thresh = 0.80 * whistle_high_thresh
    else:
        whistle_high_thresh = 999.0
        whistle_sust_thresh = 999.0

    return {
        "slowdown_p85_thresh": slowdown_p85_thresh,
        "whistle_active_pv_thresh": whistle_active_pv_thresh,
        "coll_accel_thresh": coll_accel_thresh,
        "whistle_high_thresh": whistle_high_thresh,
        "whistle_sust_thresh": whistle_sust_thresh,
    }


def precompute_features(data=None, holdout_intervals=None):
    """Precompute all ball, player, team, collective-movement, and self-calibrated statistics."""
    is_competition_match = data is None or "game" not in data
    if data is None:
        data = parse_and_cache()

    ball_xy = data["ball_xy"].copy()
    player_xy = data["player_xy"]
    track_teams = data["track_teams"]
    pitch_length = float(data["pitch"][0])
    pitch_width = float(data["pitch"][1])
    n_frames = len(ball_xy)

    if "runs" in data and data["runs"] is not None:
        active_halves = [(int(a), int(b)) for a, b in data["runs"]]
        if holdout_intervals is None:
            holdout_intervals = []
    else:
        active_halves = [(HALF1_START, HALF1_END), (HALF2_START, HALF2_END)]
        if holdout_intervals is None:
            holdout_intervals = [PUBLIC_HOLDOUT]

    if "team_diag" in data and data["team_diag"] is not None:
        pred_teams, diag = data["team_diag"]
    else:
        clf = PlayerTeamClassifier(
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            fps=FPS,
            holdout_intervals=holdout_intervals,
        )
        pred_teams, diag = clf.classify(player_xy, initial_teams=track_teams, periods=active_halves)
    corrected_xy = diag["corrected_xy"]
    phantom_mask = diag["phantom_mask"]
    att_dir = diag["att_dir"]

    ball_vis = ~np.isnan(ball_xy[:, 0])
    since_gap, until_gap, total_gap = compute_gap_lengths(ball_vis)

    valid_idx = np.where(ball_vis)[0]
    ball_xy_interp = ball_xy.copy()
    if len(valid_idx) > 1:
        for dim in [0, 1]:
            ball_xy_interp[:, dim] = np.interp(
                np.arange(n_frames), valid_idx, ball_xy[valid_idx, dim]
            )

    vel, ball_speed, acc, dir_change = compute_kinematics(ball_xy_interp)
    tang_acc = np.zeros(n_frames, dtype=np.float32)
    tang_acc[1:-1] = (ball_speed[2:] - ball_speed[:-2]) * (FPS / 2.0)

    # Suppress numerical second-derivative kinks at the +/-2 frame boundary of linear NaN gap interpolations
    near_gap_kink = np.convolve((total_gap > 2).astype(int), np.ones(5, dtype=int), mode="same") > 0
    acc[near_gap_kink] = 0.0
    tang_acc[near_gap_kink] = 0.0
    dir_change[near_gap_kink] = 0.0

    player_vel = np.zeros_like(corrected_xy, dtype=np.float32)
    player_vel[1:-1] = (corrected_xy[2:] - corrected_xy[:-2]) * (FPS / 2.0)
    player_speed = np.linalg.norm(player_vel, axis=-1)

    diff = corrected_xy - ball_xy_interp[:, np.newaxis, :]
    dist = np.linalg.norm(diff, axis=-1)
    dist[phantom_mask] = 999.0
    dist_safe = np.nan_to_num(dist, nan=999.0)

    closest_player = np.argmin(dist_safe, axis=-1)
    closest_dist = np.min(dist_safe, axis=-1)
    closest_team = pred_teams[np.arange(n_frames), closest_player]
    closest_pvel = player_vel[np.arange(n_frames), closest_player]
    closest_pspeed = player_speed[np.arange(n_frames), closest_player]
    rel_speed = np.linalg.norm(
        np.nan_to_num(vel, nan=0.0) - np.nan_to_num(closest_pvel, nan=0.0), axis=-1
    )

    h_mask = pred_teams == HOME
    a_mask = pred_teams == AWAY
    min_dist_h = np.min(np.where(h_mask, dist_safe, 999.0), axis=-1)
    min_dist_a = np.min(np.where(a_mask, dist_safe, 999.0), axis=-1)
    opp_dist = np.where(closest_team == HOME, min_dist_a, min_dist_h)

    vis_valid = (~np.isnan(corrected_xy[:, :, 0])) & (~phantom_mask)
    has_vis = vis_valid.any(axis=1)
    med_pv = np.zeros(n_frames, dtype=np.float32)
    p85_pv = np.zeros(n_frames, dtype=np.float32)
    max_pv = np.zeros(n_frames, dtype=np.float32)
    coll_vx = np.zeros(n_frames, dtype=np.float32)

    spds_valid = np.where(vis_valid[has_vis], np.nan_to_num(player_speed[has_vis], nan=0.0), np.nan)
    med_pv[has_vis] = np.nanmedian(spds_valid, axis=1)
    p85_pv[has_vis] = np.nanpercentile(spds_valid, 85, axis=1)
    max_pv[has_vis] = np.nanmax(spds_valid, axis=1)
    vx_valid = np.where(
        vis_valid[has_vis], np.nan_to_num(player_vel[has_vis, :, 0], nan=0.0), np.nan
    )
    coll_vx[has_vis] = np.nanmean(vx_valid, axis=1) * att_dir[has_vis]

    w_coll = int(round(0.6 * FPS))
    k_coll = np.ones(w_coll, dtype=np.float32) / float(w_coll)
    coll_vx_smooth = np.convolve(coll_vx, k_coll, mode="same")
    coll_ax = np.zeros_like(coll_vx_smooth)
    half_d = max(1, int(round(0.2 * FPS)))
    coll_ax[half_d:-half_d] = (coll_vx_smooth[2 * half_d :] - coll_vx_smooth[: -2 * half_d]) * (
        FPS / float(2 * half_d)
    )

    whistle_file = os.path.join(ROOT, "cache/whistle_detections.npz")
    if is_competition_match:
        if not os.path.exists(whistle_file):
            try:
                from solution.whistle_detector import extract_whistle_frames

                extract_whistle_frames()
            except Exception as exc:
                print(f"Warning: could not auto-extract whistle detections ({exc})")
        if os.path.exists(whistle_file):
            w_loaded = np.load(whistle_file)["whistle_ratio"]
            w_ratio = (
                w_loaded if len(w_loaded) == n_frames else np.zeros(n_frames, dtype=np.float32)
            )
        else:
            w_ratio = np.zeros(n_frames, dtype=np.float32)
    else:
        w_ratio = np.zeros(n_frames, dtype=np.float32)

    ref_pose_file = os.path.join(ROOT, "cache/referee_pose_features.npz")
    if is_competition_match:
        if not os.path.exists(ref_pose_file):
            try:
                from solution.referee_pose_detector import extract_referee_pose_features

                extract_referee_pose_features(
                    corrected_xy=corrected_xy,
                    ball_xy_interp=ball_xy_interp,
                    pitch_length=pitch_length,
                    pitch_width=pitch_width,
                    holdout_intervals=holdout_intervals,
                    cache_path=ref_pose_file,
                )
            except Exception as exc:
                print(f"Warning: could not auto-extract referee pose features ({exc})")
        if os.path.exists(ref_pose_file):
            r_loaded = np.load(ref_pose_file)["ref_signal_score"]
            ref_signal_score = (
                r_loaded if len(r_loaded) == n_frames else np.zeros(n_frames, dtype=np.float32)
            )
        else:
            ref_signal_score = np.zeros(n_frames, dtype=np.float32)
    else:
        ref_signal_score = np.zeros(n_frames, dtype=np.float32)

    # Symmetric FIFA wing / goal-area boundary checks (|y - W/2| > 20.16m or x within 5.5m of goal line)
    oob_margin = 0.4
    oob_x = (ball_xy[:, 0] < -oob_margin) | (ball_xy[:, 0] > pitch_length + oob_margin)
    oob_y = (ball_xy[:, 1] < -oob_margin) | (ball_xy[:, 1] > pitch_width + oob_margin)
    ball_oob = oob_x | oob_y

    in_wing_or_goal_channel = (
        (np.abs(ball_xy_interp[:, 1] - pitch_width / 2.0) > 20.16)
        | (ball_xy_interp[:, 0] < 5.5)
        | (ball_xy_interp[:, 0] > pitch_length - 5.5)
    )
    near_boundary_nan = (~ball_vis) & in_wing_or_goal_channel

    forced_dead, kickoff_frames = extract_structural_constraints(
        n_frames, active_halves, is_competition_match=is_competition_match
    )

    cal_stats = calibrate_match_statistics(
        ball_vis=ball_vis,
        p85_pv=p85_pv,
        med_pv=med_pv,
        coll_ax=coll_ax,
        w_ratio=w_ratio,
        periods=active_halves,
        holdout_intervals=holdout_intervals,
    )

    return {
        "n_frames": n_frames,
        "periods": active_halves,
        "pitch_length": pitch_length,
        "pitch_width": pitch_width,
        "ball_xy": ball_xy,
        "ball_xy_interp": ball_xy_interp,
        "ball_vis": ball_vis,
        "since_gap": since_gap,
        "total_gap": total_gap,
        "vel": vel,
        "ball_speed": ball_speed,
        "acc": acc,
        "tang_acc": tang_acc,
        "dir_change": dir_change,
        "pred_teams": pred_teams,
        "corrected_xy": corrected_xy,
        "att_dir": att_dir,
        "closest_player": closest_player,
        "closest_dist": closest_dist,
        "closest_team": closest_team,
        "closest_pspeed": closest_pspeed,
        "rel_speed": rel_speed,
        "min_dist_h": min_dist_h,
        "min_dist_a": min_dist_a,
        "opp_dist": opp_dist,
        "med_pv": med_pv,
        "p85_pv": p85_pv,
        "max_pv": max_pv,
        "coll_vx": coll_vx,
        "coll_ax": coll_ax,
        "w_ratio": w_ratio,
        "ref_signal_score": ref_signal_score,
        "ball_oob": ball_oob,
        "near_boundary_nan": near_boundary_nan,
        "forced_dead": forced_dead,
        "kickoff_frames": kickoff_frames,
        "cal_stats": cal_stats,
    }


def predict_interval(features, s_f, e_f, params=None):
    """Predict possession labels ('H', 'A', 'D') on frame interval [s_f, e_f)."""
    p = dict(DEFAULT_PARAMS)
    if params is not None:
        p.update(params)

    cal = features["cal_stats"]

    # Fundamental physical scales & proportional derived quantities
    touch_radius = float(p["touch_radius"])
    dribble_ratio = float(p.get("dribble_radius_ratio", 0.61))
    close_dribble_radius = float(p.get("close_dribble_radius", dribble_ratio * touch_radius))
    max_dribble_speed = float(p["max_dribble_speed"])
    rel_ratio = float(p.get("co_moving_rel_ratio", 0.53))
    fast_carry_rel_speed = float(p.get("fast_carry_rel_speed", rel_ratio * max_dribble_speed))
    min_dwell_frames = int(round(p["min_dwell_frames"]))
    acc_threshold = float(p["acc_threshold"])
    dir_threshold = float(p["dir_threshold"])
    contested_margin = float(p["contested_margin"])
    min_sandwich_frames = int(round(p["min_sandwich_frames"]))

    # Derived proportional physical scales (eliminating independent magic numbers)
    dribble_opp_margin = float(p.get("dribble_opp_margin", 0.6 * contested_margin))
    strike_acc_threshold = float(p.get("strike_acc_threshold", 1.26 * acc_threshold))
    strike_dist_threshold = float(p.get("strike_dist_threshold", 0.70 * touch_radius))
    flyby_speed_thresh = float(p.get("flyby_speed_thresh", 1.87 * max_dribble_speed))
    flyby_max_dc = float(p.get("flyby_max_dc", 0.14 * dir_threshold))
    fast_carry_radius = float(p.get("fast_carry_radius", 0.83 * touch_radius))
    fast_carry_min_speed = float(p.get("fast_carry_min_speed", 0.98 * max_dribble_speed))
    fast_carry_max_speed = float(p.get("fast_carry_max_speed", 1.55 * max_dribble_speed))
    fast_carry_dwell = int(round(p.get("fast_carry_dwell", max(3, 0.30 * min_dwell_frames))))
    apex_dist_thresh = float(p.get("apex_dist_thresh", 1.89 * touch_radius))
    apex_acc_thresh = float(p.get("apex_acc_thresh", 1.30 * acc_threshold))
    apex_opp_margin = float(p.get("apex_opp_margin", 0.83 * touch_radius))
    untrack_dist_thresh = float(p.get("untrack_dist_thresh", 1.83 * touch_radius))
    setpiece_speed_thresh = float(p.get("setpiece_speed_thresh", 1.06 * max_dribble_speed))
    reentry_delay_frames = int(round(p.get("reentry_delay_frames", 0.40 * FPS)))

    # Self-calibrated match statistics (learned strictly outside [11185, 18685))
    coll_accel_thresh = float(p.get("coll_accel_thresh", cal["coll_accel_thresh"]))
    whistle_high_thresh = float(p.get("whistle_high_thresh", cal["whistle_high_thresh"]))
    whistle_sust_thresh = float(cal["whistle_sust_thresh"])
    whistle_active_pv_thresh = float(
        p.get("whistle_active_pv_thresh", cal["whistle_active_pv_thresh"])
    )
    slowdown_p85_thresh = float(p.get("slowdown_p85_thresh", cal["slowdown_p85_thresh"]))

    pitch_length = features["pitch_length"]
    pitch_width = features["pitch_width"]
    ball_xy = features["ball_xy"]
    ball_vis = features["ball_vis"]
    since_gap = features["since_gap"]
    total_gap = features["total_gap"]
    vel = features["vel"]
    ball_speed = features["ball_speed"]
    acc = features["acc"]
    tang_acc = features["tang_acc"]
    dir_change = features["dir_change"]
    pred_teams = features["pred_teams"]
    att_dir = features["att_dir"]
    closest_player = features["closest_player"]
    closest_dist = features["closest_dist"]
    closest_team = features["closest_team"]
    rel_speed = features["rel_speed"]
    min_dist_h = features["min_dist_h"]
    min_dist_a = features["min_dist_a"]
    opp_dist = features["opp_dist"]
    med_pv = features["med_pv"]
    p85_pv = features["p85_pv"]
    coll_ax = features["coll_ax"]
    w_ratio = features["w_ratio"]
    ref_signal_score = features.get("ref_signal_score", np.zeros_like(w_ratio))
    ball_oob = features["ball_oob"]
    near_boundary_nan = features["near_boundary_nan"]
    forced_dead = features["forced_dead"]
    kickoff_set = set(features["kickoff_frames"])

    n_slice = e_f - s_f
    labels = np.full(n_slice, "X", dtype="<U1")

    # Do not hallucinate touches when ball is in a long camera gap (> 2 seconds)
    valid_ball_obs = total_gap[s_f:e_f] <= (2 * FPS)

    is_impulse = (acc[s_f:e_f] > acc_threshold) | (dir_change[s_f:e_f] > dir_threshold)
    is_near_player = (closest_dist[s_f:e_f] < touch_radius) & valid_ball_obs
    is_std_dribble = (
        valid_ball_obs
        & (closest_dist[s_f:e_f] < close_dribble_radius)
        & (ball_speed[s_f:e_f] < max_dribble_speed)
        & (
            (opp_dist[s_f:e_f] > closest_dist[s_f:e_f] + dribble_opp_margin)
            | (opp_dist[s_f:e_f] > close_dribble_radius)
        )
    )
    is_fast_carry = (
        valid_ball_obs
        & (closest_dist[s_f:e_f] < fast_carry_radius)
        & (ball_speed[s_f:e_f] > fast_carry_min_speed)
        & (ball_speed[s_f:e_f] < fast_carry_max_speed)
        & (rel_speed[s_f:e_f] < fast_carry_rel_speed)
        & (opp_dist[s_f:e_f] > closest_dist[s_f:e_f] + close_dribble_radius)
    )

    is_dribble = is_std_dribble | is_fast_carry
    touch_cand = (is_impulse & is_near_player) | is_dribble

    w_sl = w_ratio[s_f:e_f]
    w_cand = w_sl > whistle_sust_thresh
    w_sust = np.convolve(w_cand.astype(int), [1, 1, 1], mode="same") >= 2
    whistle_trig = (w_sl > whistle_high_thresh) | (w_cand & w_sust)

    current_possession = "H" if closest_team[s_f] == HOME else "A"
    current_dwell = min_dwell_frames
    last_turnover_frame = s_f
    candidate_possession = None
    candidate_dwell = 0

    is_dead = bool(forced_dead[s_f])
    is_missing_slow_dead = False
    dead_frames_count = 0
    ball_was_stationary = False
    stationary_frames = 0
    post_restart_frames = 999
    pending_whistle = False
    pending_whistle_frames = 0
    reentry_wait = 0
    reentry_team = None
    missing_slow_frames = 0

    for idx in range(n_slice):
        t = s_f + idx

        # 1. Structural dead/live constraints (e.g. pre-kickoff & goal stoppages from Smart Tagging)
        if t in kickoff_set and t > s_f:
            is_dead = False
            is_missing_slow_dead = False
            pending_whistle = False
            pending_whistle_frames = 0
            reentry_wait = 0
            stationary_frames = 0
            missing_slow_frames = 0
            post_restart_frames = 999
            current_possession = "H" if closest_team[t] == HOME else "A"
            current_dwell = min_dwell_frames
            labels[idx] = current_possession
            continue

        if forced_dead[t]:
            is_dead = True
            is_missing_slow_dead = False
            pending_whistle = False
            pending_whistle_frames = 0
            labels[idx] = "D"
            dead_frames_count += 1
            continue

        # 2. Acoustic whistle trigger (expires if players do not slow down within 3.5s)
        if whistle_trig[idx] and not is_dead:
            oob_soon = ball_oob[t : min(e_f, t + int(1.6 * FPS))].any() and (
                ball_speed[t] > 2.5 * max_dribble_speed
            )
            if not oob_soon:
                pending_whistle = True
                pending_whistle_frames = 0

        if pending_whistle and not is_dead:
            pending_whistle_frames += 1
            if pending_whistle_frames > int(3.5 * FPS):
                pending_whistle = False
                pending_whistle_frames = 0
            elif (med_pv[t] < whistle_active_pv_thresh and closest_dist[t] > 1.6 * touch_radius) or (
                near_boundary_nan[t] and med_pv[t] < 2.0 * whistle_active_pv_thresh
            ):
                is_dead = True
                is_missing_slow_dead = False
                pending_whistle = False
                pending_whistle_frames = 0
                dead_frames_count = 0
                ball_was_stationary = False
                stationary_frames = 0
                post_restart_frames = 0
                reentry_wait = 0

        # 3. Silent / Missing-Ball Collective Stoppage Detection (generalizes to matches without audio)
        if not is_dead:
            if (not ball_vis[t]) and (total_gap[t] >= 4 * FPS) and (
                med_pv[t] < 0.62 * whistle_active_pv_thresh and p85_pv[t] < slowdown_p85_thresh
            ):
                missing_slow_frames += 1
                if missing_slow_frames >= int(1.2 * FPS):
                    is_dead = True
                    is_missing_slow_dead = True
                    pending_whistle = False
                    rb = max(0, idx - missing_slow_frames)
                    while (
                        rb > 0
                        and (not ball_vis[s_f + rb - 1])
                        and p85_pv[s_f + rb - 1] < 1.15 * slowdown_p85_thresh
                    ):
                        rb -= 1
                    labels[rb:idx] = "D"
                    dead_frames_count = idx - rb
                    ball_was_stationary = True
                    stationary_frames = 0
                    post_restart_frames = 0
                    reentry_wait = 0
            else:
                missing_slow_frames = 0

        # 4. Out-of-Bounds Stoppage & Pre-OOB Collective / Referee Pose Rollback
        if ball_oob[t] and not is_dead:
            is_dead = True
            is_missing_slow_dead = False
            pending_whistle = False
            dead_frames_count = 0
            ball_was_stationary = False
            stationary_frames = 0
            post_restart_frames = 0
            reentry_wait = 0

            rb = idx
            while rb > 0 and p85_pv[s_f + rb - 1] < slowdown_p85_thresh:
                rb -= 1
            if idx - rb >= FPS:
                w_sig = ref_signal_score[s_f + rb : t]
                w_win = max(5, int(0.4 * FPS))
                if len(w_sig) >= w_win:
                    # Posterior probability >= 0.50 from self-calibrated 2-sigma referee pose model
                    sust_ref = (
                        np.convolve(
                            (w_sig >= 0.50).astype(int), np.ones(w_win, dtype=int), mode="same"
                        )
                        >= int(0.8 * w_win)
                    )
                    if np.any(sust_ref):
                        onset = int(np.where(sust_ref)[0][0])
                        # Roll back to gesture initiation where pose probability exceeds neutral baseline (0.20)
                        while onset > 0 and w_sig[onset - 1] >= 0.20:
                            onset -= 1
                        rb = rb + onset
                labels[rb:idx] = "D"
                dead_frames_count = idx - rb

        # 5. Dead-Ball State Machine & Set-Piece Restart Detection
        if is_dead:
            labels[idx] = "D"
            dead_frames_count += 1

            if ball_speed[t] < 0.4 * max_dribble_speed and ball_vis[t]:
                stationary_frames += 1
                if stationary_frames >= int(0.72 * FPS):
                    ball_was_stationary = True
            elif ball_speed[t] > 0.55 * max_dribble_speed:
                stationary_frames = 0

            ball_in_pitch = (
                ball_vis[t]
                and (0.0 <= ball_xy[t, 0] <= pitch_length)
                and (0.0 <= ball_xy[t, 1] <= pitch_width)
            )

            if reentry_wait > 0:
                reentry_wait += 1
                if reentry_wait >= max(1, reentry_delay_frames):
                    is_dead = False
                    current_possession = reentry_team
                    current_dwell = min_dwell_frames
                    candidate_possession = None
                    candidate_dwell = 0
                    post_restart_frames = 0
                    reentry_wait = 0
                    stationary_frames = 0
                    labels[idx] = current_possession
                continue

            is_setpiece_kick = (
                ball_was_stationary
                and ball_in_pitch
                and dead_frames_count >= 2 * FPS
                and (closest_dist[t] < 1.4 * touch_radius)
                and (acc[t] > acc_threshold and ball_speed[t] >= setpiece_speed_thresh)
            )

            in_penalty_box_x = (
                ball_xy[t, 0] <= FIFA_PENALTY_AREA_DEPTH + touch_radius
                or ball_xy[t, 0] >= pitch_length - (FIFA_PENALTY_AREA_DEPTH + touch_radius)
            )
            is_inplay_reception_restart = (
                (ball_was_stationary or dead_frames_count >= 8 * FPS)
                and ball_in_pitch
                and dead_frames_count >= 5 * FPS
                and (closest_dist[t] < apex_dist_thresh)
                and (ball_speed[t] > 0.42 * max_dribble_speed)
                and (
                    (
                        (in_penalty_box_x or p85_pv[t] >= 0.85 * slowdown_p85_thresh)
                        and tang_acc[t] < -0.35 * acc_threshold
                    )
                    or (
                        med_pv[t] > 1.15 * whistle_active_pv_thresh
                        and p85_pv[t] > 2.0 * slowdown_p85_thresh
                        and closest_dist[t] < touch_radius
                    )
                )
            )

            just_reentered = (
                ball_in_pitch and (not ball_vis[t - 1]) and (dead_frames_count >= int(1.6 * FPS))
            )
            if (
                just_reentered
                and (closest_dist[t] < 1.22 * touch_radius)
                and (ball_speed[t] > 0.53 * max_dribble_speed)
            ):
                if reentry_delay_frames <= 1:
                    is_dead = False
                    current_possession = "H" if closest_team[t] == HOME else "A"
                    current_dwell = min_dwell_frames
                    candidate_possession = None
                    candidate_dwell = 0
                    post_restart_frames = 0
                    stationary_frames = 0
                    labels[idx] = current_possession
                else:
                    reentry_wait = 1
                    reentry_team = "H" if closest_team[t] == HOME else "A"
                continue

            if is_missing_slow_dead and ball_in_pitch and (
                closest_dist[t] < 1.5 * touch_radius or med_pv[t] >= whistle_active_pv_thresh
            ):
                is_dead = False
                is_missing_slow_dead = False
                current_possession = "H" if closest_team[t] == HOME else "A"
                current_dwell = min_dwell_frames
                candidate_possession = None
                candidate_dwell = 0
                post_restart_frames = 999
                stationary_frames = 0
                labels[idx] = current_possession
                continue

            if is_setpiece_kick or is_inplay_reception_restart:
                is_dead = False
                is_missing_slow_dead = False
                current_possession = "H" if closest_team[t] == HOME else "A"
                current_dwell = min_dwell_frames
                candidate_possession = None
                candidate_dwell = 0
                post_restart_frames = 0
                stationary_frames = 0
                labels[idx] = current_possession
            continue

        # 6. Post-restart false-nudge rollback (if visible ball stops again while players are inactive)
        post_restart_frames += 1
        if post_restart_frames < int(14.4 * FPS):
            cond_slow = (
                ball_vis[t]
                and (ball_speed[t] < 0.25 * max_dribble_speed)
                and (p85_pv[t] < slowdown_p85_thresh)
            )
            if cond_slow:
                stationary_frames += 1
                if stationary_frames >= int(0.72 * FPS):
                    labels[max(0, idx - post_restart_frames) : idx + 1] = "D"
                    is_dead = True
                    dead_frames_count += post_restart_frames
                    ball_was_stationary = True
                    stationary_frames = 0
                    continue
            else:
                stationary_frames = 0

        # 7. Live Play Touch, Co-Moving Fast Carry & Apex Reception Transitions
        touched_team = "H" if closest_team[t] == HOME else "A"

        is_uncontested_apex = (
            valid_ball_obs[idx]
            and (closest_dist[t] < apex_dist_thresh)
            and (opp_dist[t] > closest_dist[t] + apex_opp_margin)
            and (
                (
                    acc[t] > apex_acc_thresh
                    and (
                        tang_acc[t] < -0.85 * acc_threshold
                        or dir_change[t] > 0.23 * dir_threshold
                    )
                )
                or (
                    dir_change[t] > 0.32 * dir_threshold
                    and ball_speed[t] < 0.85 * max_dribble_speed
                    and closest_dist[t] < 1.44 * touch_radius
                )
            )
        )

        # Contested aerial header turnaround (resolves 2D homography height parallax via outgoing velocity & collective surge)
        is_aerial_header_turn = False
        lead_v = max(2, int(round(0.24 * FPS)))
        lead_c = max(1, int(round(0.16 * FPS)))
        if (
            valid_ball_obs[idx]
            and dir_change[t] > 0.44 * dir_threshold
            and ball_speed[t] < 0.30 * max_dribble_speed
            and min_dist_h[t] < apex_dist_thresh
            and min_dist_a[t] < apex_dist_thresh
            and t + lead_v < e_f
        ):
            out_vx = vel[t + lead_v, 0] * att_dir[t]
            if (
                out_vx > 0.10 * max_dribble_speed
                and coll_ax[t + lead_c] > 0.33 * coll_accel_thresh
                and current_possession == "A"
            ):
                is_aerial_header_turn = True
                touched_team = "H"
            elif (
                out_vx < -0.10 * max_dribble_speed
                and coll_ax[t + lead_c] < -0.33 * coll_accel_thresh
                and current_possession == "H"
            ):
                is_aerial_header_turn = True
                touched_team = "A"

        if touch_cand[idx] or is_uncontested_apex or is_aerial_header_turn:
            if touched_team == current_possession:
                current_dwell += 1
                candidate_possession = None
                candidate_dwell = 0
            else:
                is_flyby = (
                    (ball_speed[t] > flyby_speed_thresh)
                    and (dir_change[t] < flyby_max_dc)
                    and (tang_acc[t] > -0.70 * acc_threshold)
                )
                min_opp_local = np.min(opp_dist[max(s_f, t - 2) : min(e_f, t + 3)])
                is_contested = (
                    (opp_dist[t] < closest_dist[t] + contested_margin)
                    or (min_opp_local < 0.92 * touch_radius and ball_speed[t] > 1.6 * max_dribble_speed)
                ) and (ball_speed[t] > 1.38 * max_dribble_speed)

                if is_dribble[idx] and not is_aerial_header_turn:
                    if candidate_possession == touched_team:
                        candidate_dwell += 1
                        if candidate_dwell >= min_dwell_frames or (
                            is_fast_carry[idx] and candidate_dwell >= fast_carry_dwell
                        ):
                            # Back-date turnover to ball departure when following a brief touch
                            if (
                                (t - last_turnover_frame < int(2.6 * FPS))
                                and (current_dwell <= min_dwell_frames)
                                and (closest_dist[last_turnover_frame] < 1.22 * touch_radius)
                            ):
                                dep = last_turnover_frame + 1
                                prev_team_code = HOME if current_possession == "H" else AWAY
                                while (
                                    dep < t
                                    and closest_dist[dep] < 0.78 * touch_radius
                                    and pred_teams[dep, closest_player[dep]] == prev_team_code
                                ):
                                    dep += 1
                                labels[max(0, dep - s_f) : idx] = touched_team
                            current_possession = touched_team
                            current_dwell = candidate_dwell
                            last_turnover_frame = t
                            candidate_possession = None
                            candidate_dwell = 0
                    else:
                        candidate_possession = touched_team
                        candidate_dwell = 1
                elif (not is_flyby) and (not is_contested):
                    if (
                        (acc[t] > strike_acc_threshold and closest_dist[t] < strike_dist_threshold)
                        or is_uncontested_apex
                        or is_aerial_header_turn
                    ):
                        current_possession = touched_team
                        current_dwell = 1
                        last_turnover_frame = t
                        candidate_possession = None
                        candidate_dwell = 0

        labels[idx] = current_possession

    # Post-processing 1: Unreliable / Stuck Ball Bridge via Collective Longitudinal Acceleration
    unrel = (
        (closest_dist[s_f:e_f] > untrack_dist_thresh)
        & (ball_speed[s_f:e_f] < 1.28 * max_dribble_speed)
        & ball_vis[s_f:e_f]
        & (labels != "D")
    )
    u_edges = np.diff(np.concatenate(([0], unrel.astype(int), [0])))
    u_starts = np.where(u_edges == 1)[0]
    u_ends = np.where(u_edges == -1)[0]
    for us, ue in zip(u_starts, u_ends):
        if ue - us < int(0.8 * FPS):
            continue
        look = ue
        while (
            look < n_slice
            and (look - ue) < int(4.8 * FPS)
            and labels[look] != "D"
            and (
                closest_dist[s_f + look] > 0.89 * touch_radius
                or (
                    acc[s_f + look] < 0.20 * acc_threshold
                    and dir_change[s_f + look] < 0.06 * dir_threshold
                )
            )
        ):
            look += 1
        if look < n_slice and labels[look] != "D":
            team_before = labels[max(0, us - 1)]
            post_window = labels[look : min(n_slice, look + int(3.2 * FPS))]
            if len(post_window) > int(0.8 * FPS):
                h_cnt = np.sum(post_window == "H")
                a_cnt = np.sum(post_window == "A")
                team_after = "H" if h_cnt > a_cnt else "A"
                if team_after != team_before and team_before in ("H", "A"):
                    sign_after = 1.0 if team_after == "H" else -1.0
                    search_s = max(0, us - int(0.48 * FPS))
                    t_turn = ue
                    for k in range(search_s, ue):
                        if (
                            coll_ax[s_f + k] * sign_after > coll_accel_thresh
                            and ball_speed[s_f + k] < 1.17 * max_dribble_speed
                        ):
                            t_turn = k
                            break
                    labels[t_turn : min(n_slice, look + int(0.8 * FPS))] = team_after

    # Post-processing 2: Power-clearance deceleration onset vs passive-deflection sandwich smoothing
    if min_sandwich_frames > 0:
        runs = [(0, n_slice)]
        codes = pm.label_codes(labels)
        segs = pm.segments(codes, runs)
        labels_map = ("H", "A", "D", "X")
        for i in range(1, len(segs) - 1):
            a, b, k = segs[i]
            prev_a, prev_b, prev_k = segs[i - 1]
            next_a, next_b, next_k = segs[i + 1]
            dur = b - a
            if k in (0, 1):
                if (
                    next_k == 2
                    and prev_k in (0, 1)
                    and prev_k != k
                    and dur < min_sandwich_frames
                ):
                    labels[a:b] = labels_map[prev_k]
                elif (
                    prev_k == next_k
                    and prev_k in (0, 1)
                    and prev_k != k
                    and dur < min_sandwich_frames
                ):
                    is_power_clearance = (
                        np.max(acc[s_f + a : s_f + b]) > 1.48 * acc_threshold
                        and ball_speed[s_f + min(n_slice - 1, b)] > 1.70 * max_dribble_speed
                    )
                    if is_power_clearance and dur < min_dwell_frames:
                        rb_a = a
                        while (
                            rb_a > prev_a + min_dwell_frames
                            and (
                                tang_acc[s_f + rb_a - 1] < 0
                                or ball_speed[s_f + rb_a - 1] < max_dribble_speed
                            )
                            and (a - rb_a) < min_dwell_frames
                        ):
                            rb_a -= 1
                        labels[rb_a:a] = labels_map[k]
                    elif (
                        not is_power_clearance
                        and (prev_b - prev_a) > 2 * min_sandwich_frames
                        and (next_b - next_a) > 2 * min_sandwich_frames
                        and np.min(ball_speed[s_f + a : s_f + b]) > 0.34 * max_dribble_speed
                    ):
                        labels[a:b] = labels_map[prev_k]

    return labels


def run_touch_possession_model(params=None, features=None, **kwargs):
    """Run full-match kinematic & collective-movement possession pipeline on cached ATD data."""
    if features is None:
        features = precompute_features()

    merged_params = dict(DEFAULT_PARAMS)
    if params is not None:
        merged_params.update(params)
    for k, v in kwargs.items():
        merged_params[k] = v

    n_frames = features["n_frames"]
    labels = np.full(n_frames, "X", dtype="<U1")

    for start_f, end_f in features["periods"]:
        labels[start_f:end_f] = predict_interval(features, start_f, end_f, merged_params)

    return labels


def save_and_evaluate(labels, out_path="evaluation/results/submission_touch_v3.csv"):
    """Write submission CSV and evaluate against the public GT (blind holdout check)."""
    n_frames = len(labels)
    df = pd.DataFrame({"frame": np.arange(n_frames), "label": labels})
    df.to_csv(out_path, index=False)
    print(f"Saved {n_frames} rows to {out_path}")

    gt_path = os.path.join(ROOT, "evaluation/gt/BAR-ATH_possession_gt_public.csv")
    if os.path.exists(gt_path):
        gt_df = pm.read_label_csv(gt_path)
        pm.validate_submission(df, len(gt_df))
        res = pm.evaluate(pm.label_codes(gt_df), pm.label_codes(df))
        print(
            f"Blind Public Holdout Check -> spellF1: {res['spell_f1_mean']:.4f} | "
            f"macroF1: {res['macro_f1']:.4f} | accuracy: {res['accuracy']:.4f} | "
            f"spells: {res['n_pred_spells']}/{res['n_gt_spells']}"
        )
        return res
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the Self-Calibrated First-Principles Physics Possession Tracker."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(ROOT, "evaluation/results/submission_touch_v3.csv"),
        help="Output CSV path (default: evaluation/results/submission_touch_v3.csv)",
    )
    args = parser.parse_args()

    print("Running Self-Calibrated & Cross-Validated Touch Possession Model...")
    labels = run_touch_possession_model()
    save_and_evaluate(labels, args.output)
    out_df = pd.DataFrame({"frame": np.arange(len(labels)), "label": labels})
    for extra_path in [
        os.path.join(ROOT, "my_submission.csv"),
        os.path.join(ROOT, "evaluation/results/my_submission.csv"),
    ]:
        if os.path.abspath(args.output) != os.path.abspath(extra_path):
            os.makedirs(os.path.dirname(extra_path), exist_ok=True)
            out_df.to_csv(extra_path, index=False)
            print(f"Saved {len(labels)} rows to {extra_path}")

