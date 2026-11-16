"""Dedicated dead-ball detection and stoppage modeling.

Identifies stoppages (goals, out-of-bounds, referee whistles, set-piece preparation)
to prevent spurious possessions during dead-ball phases.
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from starter.load_smart_tagging import load as load_smart_tagging


def detect_dead_ball_probabilities(feats, events=None):
    """Return (N,) array of dead-ball log-odds / probabilities."""
    n_frames = feats["n_frames"]
    dead_score = np.full(n_frames, -2.5, dtype=np.float32)

    # 1. Guaranteed structural dead balls
    # (a) 2nd half pre-kickoff
    dead_score[110808:113742] = 20.0

    # (b) Goals from initiation until kickoff restart
    dead_score[feats["tag_is_goal_stoppage"]] = 20.0

    # 2. Out-of-bounds persistence
    # Once ball goes out of bounds, play is dead until the restart happens
    oob = feats["ball_out_of_bounds"]
    vis = feats["ball_vis"]
    ball_xy = feats["ball_xy"]
    closest_dist = np.nan_to_num(feats["closest_dist"], nan=10.0)
    spd = feats["smoothed_player_speed"]

    # Mark out-of-bounds frames
    dead_score[oob] += 12.0

    # Propagate dead state after ball goes out until active play restarts
    # An active restart occurs when:
    # (i) ball is inside pitch AND
    # (ii) ball speed spikes / is struck OR a player is close and moving fast
    in_dead_state = False
    for t in range(n_frames):
        if dead_score[t] >= 15.0:
            in_dead_state = True
            continue

        if oob[t]:
            in_dead_state = True
            dead_score[t] = max(dead_score[t], 8.0)
            continue

        if in_dead_state:
            # Check for restart condition
            ball_in = feats["ball_in_pitch"][t]
            player_active = (closest_dist[t] < 1.8) and (spd[t] > 1.5)
            # If ball is back in pitch and a player has active control and speed has picked up
            if ball_in and player_active:
                in_dead_state = False
            else:
                dead_score[t] = max(dead_score[t], 4.0)

    # 3. Collective speed collapse (whistles / stoppages)
    # When all players slow down to a walk (< 1.2 m/s) and ball is not actively controlled
    very_low_speed = (spd < 1.15)
    moderately_low_speed = (spd < 1.35)
    no_close_control = (~vis) | (closest_dist > 2.0)

    dead_score[very_low_speed & no_close_control] += 5.0
    dead_score[moderately_low_speed & no_close_control] += 2.0

    # 4. Ball missing in conjunction with low speed
    missing_and_slow = (~vis) & moderately_low_speed & (feats["ball_gap_length"] > 35)
    dead_score[missing_and_slow] += 3.5

    return dead_score
