"""Feature extraction for frame-level possession modeling.

Computes geometric proximity, kinematics, ball status, collective player dynamics,
and Smart Tagging priors over all 199,319 frames in < 1 second.
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from starter.load_smart_tagging import load as load_smart_tagging
from solution.parse_atd import load_cached_atd

HOME = 0
AWAY = 1
DEAD = 2
NONE = -1


def compute_features(atd=None, events=None):
    """Compute and return dictionary of frame-level features."""
    if atd is None:
        atd = load_cached_atd()
    if events is None:
        events = load_smart_tagging()

    ball_xy = atd["ball_xy"]              # (N, 2)
    ball_speed = atd["ball_speed"]        # (N,)
    player_xy = atd["player_xy"]          # (N, 46, 2)
    player_speed = atd["player_speed"]    # (N, 46)
    track_teams = atd["track_teams"]      # (46,)
    pitch_len, pitch_wid = atd["pitch"]
    n_frames = len(ball_xy)

    home_mask = (track_teams == HOME)
    away_mask = (track_teams == AWAY)

    # 1. Ball visibility and boundaries
    ball_vis = ~np.isnan(ball_xy[:, 0])
    ball_in_pitch = np.zeros(n_frames, dtype=bool)
    ball_in_pitch[ball_vis] = (
        (ball_xy[ball_vis, 0] >= 0.0) & (ball_xy[ball_vis, 0] <= pitch_len) &
        (ball_xy[ball_vis, 1] >= 0.0) & (ball_xy[ball_vis, 1] <= pitch_wid)
    )
    ball_out_of_bounds = ball_vis & (~ball_in_pitch)

    # Distance beyond boundary
    dist_oob = np.zeros(n_frames, dtype=np.float32)
    if np.any(ball_out_of_bounds):
        bx = ball_xy[ball_out_of_bounds, 0]
        by = ball_xy[ball_out_of_bounds, 1]
        dx = np.maximum(0.0, np.maximum(-bx, bx - pitch_len))
        dy = np.maximum(0.0, np.maximum(-by, by - pitch_wid))
        dist_oob[ball_out_of_bounds] = np.sqrt(dx * dx + dy * dy)

    # Ball gap length (consecutive frames missing)
    ball_gap_length = np.zeros(n_frames, dtype=np.int32)
    current_gap = 0
    for i in range(n_frames):
        if not ball_vis[i]:
            current_gap += 1
            ball_gap_length[i] = current_gap
        else:
            current_gap = 0

    # 2. Player-to-ball distances
    # diff: (N, 46, 2)
    diff = player_xy - ball_xy[:, None, :]
    dist_sq = diff[:, :, 0]**2 + diff[:, :, 1]**2
    # Mask missing player tracks with max pitch diagonal squared (125m^2 = 15625)
    missing_players = np.isnan(player_xy[:, :, 0])
    dist_sq[missing_players] = 15625.0

    home_dists_sq = np.min(dist_sq[:, home_mask], axis=1)
    away_dists_sq = np.min(dist_sq[:, away_mask], axis=1)

    min_dist_home = np.sqrt(home_dists_sq)
    min_dist_away = np.sqrt(away_dists_sq)
    min_dist_home = np.clip(min_dist_home, 0.0, 125.0)
    min_dist_away = np.clip(min_dist_away, 0.0, 125.0)

    min_dist_home[~ball_vis] = np.nan
    min_dist_away[~ball_vis] = np.nan

    closest_dist = np.minimum(min_dist_home, min_dist_away)
    dist_margin = min_dist_away - min_dist_home  # > 0 means Home is closer
    dist_margin = np.clip(dist_margin, -30.0, 30.0)

    # Player counts within 3m
    home_count_3m = np.sum((dist_sq[:, home_mask] <= 9.0) & (~missing_players[:, home_mask]), axis=1)
    away_count_3m = np.sum((dist_sq[:, away_mask] <= 9.0) & (~missing_players[:, away_mask]), axis=1)
    home_count_3m[~ball_vis] = 0
    away_count_3m[~ball_vis] = 0

    # 3. Ball Kinematics & Touch Detection
    dt = 0.04 # 25 fps
    v_ball = np.full_like(ball_xy, np.nan)
    v_ball[1:-1] = (ball_xy[2:] - ball_xy[:-2]) / (2 * dt)
    kinematic_ball_speed = np.linalg.norm(v_ball, axis=1)

    a_ball = np.full_like(ball_xy, np.nan)
    a_ball[1:-1] = (v_ball[2:] - v_ball[:-2]) / (2 * dt)
    acc_ball = np.linalg.norm(a_ball, axis=1)

    # Touch detection: sudden ball acceleration impulse while player is in close control
    is_touch = ball_vis & (~np.isnan(acc_ball)) & (acc_ball > 10.0) & (closest_dist <= 2.0)
    touch_home = is_touch & (min_dist_home < min_dist_away)
    touch_away = is_touch & (min_dist_away < min_dist_home)

    # Aerial duel / contested ball: opposing players both within contest radius
    contested = ball_vis & (min_dist_home <= 2.2) & (min_dist_away <= 2.2)

    # Ball in flight: moving fast without players in ground contact
    in_flight = ball_vis & (kinematic_ball_speed > 10.0) & (closest_dist > 2.0)

    # Touch momentum with exponential decay (half-life ~ 1.0s / 25 frames)
    decay = 0.97
    momentum_home = np.zeros(n_frames, dtype=np.float32)
    momentum_away = np.zeros(n_frames, dtype=np.float32)
    for t in range(1, n_frames):
        momentum_home[t] = momentum_home[t - 1] * decay + (2.5 if touch_home[t] else 0.0)
        momentum_away[t] = momentum_away[t - 1] * decay + (2.5 if touch_away[t] else 0.0)

    # 4. Collective player speeds
    with np.errstate(all='ignore'):
        mean_player_speed = np.nanmean(player_speed, axis=1)
    mean_player_speed = np.nan_to_num(mean_player_speed, nan=0.0)

    # Rolling smoothed player speed (window 25 frames = 1.0 s)
    kernel_size = 25
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    smoothed_player_speed = np.convolve(mean_player_speed, kernel, mode="same")

    # 5. Smart Tagging integration
    home_name = atd["home_name"]  # 'Barcelona'
    away_name = atd["away_name"]  # 'Atletic Club'

    tag_is_set_piece = np.zeros(n_frames, dtype=bool)
    tag_is_goal_stoppage = np.zeros(n_frames, dtype=bool)
    tag_phase_team = np.full(n_frames, NONE, dtype=np.int8)

    # Process goals and kickoff restarts
    goals = [e for e in events if e.get("code") == "GOALS"]
    restarts = [e for e in events if e.get("tags", {}).get("Type") == "Kick Off (After Goal)"]
    for i, g in enumerate(goals):
        gs = g["start_frame"]
        # Goal stoppage runs from shortly after goal initiation until kickoff restart
        re_start = restarts[i]["start_frame"] if i < len(restarts) else g["end_frame"]
        # Stoppage begins around 25 frames after goal event starts (after the ball crosses line)
        stoppage_start = min(gs + 25, re_start)
        tag_is_goal_stoppage[stoppage_start:re_start] = True

    # Process set pieces and team phases
    for ev in events:
        code = ev.get("code", "")
        sf = max(0, min(n_frames - 1, ev["start_frame"]))
        ef = max(0, min(n_frames, ev["end_frame"]))
        tags = ev.get("tags", {})
        team_str = tags.get("Team", "")

        if code == "SET PIECES":
            tag_is_set_piece[sf:ef] = True

        # Assign phase team
        team_code = NONE
        if team_str == home_name:
            team_code = HOME
        elif team_str == away_name:
            team_code = AWAY

        # In possession phases, the tagged team has the ball
        possession_codes = {
            "BUILD UP", "PROGRESSION", "ATTACKING TRANSITION",
            "BALL IN FINAL THIRD", "BALL IN THE BOX", "CREATING CHANCES"
        }
        defensive_codes = {
            "DEFENSIVE TRANSITION", "DEFENDING IN DEFENSIVE THIRD",
            "DEFENDING IN MIDDLE THIRD", "DEFENDING IN ATTACKING THIRD"
        }

        if code in possession_codes and team_code != NONE:
            tag_phase_team[sf:ef] = team_code
        elif code in defensive_codes and team_code != NONE:
            # Defending team is opposing the possessing team
            opp_code = AWAY if team_code == HOME else HOME
            tag_phase_team[sf:ef] = opp_code

    return {
        "n_frames": n_frames,
        "ball_xy": ball_xy,
        "ball_speed": ball_speed,
        "ball_vis": ball_vis,
        "ball_in_pitch": ball_in_pitch,
        "ball_out_of_bounds": ball_out_of_bounds,
        "dist_oob": dist_oob,
        "ball_gap_length": ball_gap_length,
        "min_dist_home": min_dist_home,
        "min_dist_away": min_dist_away,
        "closest_dist": closest_dist,
        "dist_margin": dist_margin,
        "is_touch": is_touch,
        "touch_home": touch_home,
        "touch_away": touch_away,
        "momentum_home": momentum_home,
        "momentum_away": momentum_away,
        "contested": contested,
        "in_flight": in_flight,
        "acc_ball": acc_ball,
        "home_count_3m": home_count_3m,
        "away_count_3m": away_count_3m,
        "mean_player_speed": mean_player_speed,
        "smoothed_player_speed": smoothed_player_speed,
        "tag_is_set_piece": tag_is_set_piece,
        "tag_is_goal_stoppage": tag_is_goal_stoppage,
        "tag_phase_team": tag_phase_team,
    }


if __name__ == "__main__":
    import time
    t0 = time.time()
    feats = compute_features()
    print(f"Features computed in {time.time() - t0:.2f}s for {feats['n_frames']} frames.")
    print(f"  Ball out of bounds frames: {np.sum(feats['ball_out_of_bounds'])}")
    print(f"  Goal stoppage frames: {np.sum(feats['tag_is_goal_stoppage'])}")
    print(f"  Tagged team phase frames: {np.sum(feats['tag_phase_team'] != NONE)}")
