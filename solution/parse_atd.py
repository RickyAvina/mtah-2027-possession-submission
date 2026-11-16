"""Fast tactical ATD parser and binary cache.

Parses BAR-ATH_tactical_atd.txt (199,319 frames) into compact numpy arrays
and caches them to cache/tactical_atd.npz for sub-second loading.
"""

import os
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "tactical_atd.npz")

HOME = 0
AWAY = 1
N_TRACKS = 46


def team_of_each_track(meta_path):
    """Array over track slots: 0 if Home, 1 if Away."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError(f"Expected 2 teams in {meta_path}, found {len(team_ids)}")
    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError(f"No player slots found in {meta_path}")
    return np.array([HOME if t == home_id else AWAY for t in slots], dtype=np.int8)


def team_names(meta_path):
    """(home_name, away_name)"""
    root = ET.parse(meta_path).getroot()
    return tuple(team.findtext("Name") for team in root.iter("Team"))


def parse_and_cache(force=False, data_dir=None):
    """Parse raw ATD text and save to compressed npz cache."""
    if os.path.exists(CACHE_FILE) and not force:
        print(f"Loading existing cache from {CACHE_FILE} ...")
        return load_cached_atd()

    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, data_dir)
    n_frames = source_frames("tactical", GAME)
    pitch_length, pitch_width = GAMES[GAME]["pitch"]
    track_teams = team_of_each_track(meta_path)
    home_name, away_name = team_names(meta_path)

    print(f"Parsing ATD: {raw_path}")
    print(f"Frames: {n_frames}, Home: {home_name}, Away: {away_name}, Pitch: {pitch_length}x{pitch_width}m")

    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)
    ball_speed = np.full(n_frames, np.nan, dtype=np.float32)
    player_xy = np.full((n_frames, N_TRACKS, 2), np.nan, dtype=np.float32)
    player_speed = np.full((n_frames, N_TRACKS), np.nan, dtype=np.float32)

    t0 = time.time()
    with open(raw_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":")
            if len(parts) != 3:
                continue

            frame_id = int(parts[0]) - 1  # 0-indexed video frame
            if frame_id < 0 or frame_id >= n_frames:
                continue

            # Parse ball
            ball_str = parts[2]
            if ball_str and not ball_str.startswith("NaN"):
                b_items = ball_str.split(",")
                if len(b_items) >= 2 and b_items[0] != "NaN" and b_items[1] != "NaN":
                    ball_xy[frame_id, 0] = float(b_items[0]) * pitch_length
                    ball_xy[frame_id, 1] = float(b_items[1]) * pitch_width
                if len(b_items) >= 3 and b_items[2] != "NaN":
                    ball_speed[frame_id] = float(b_items[2])

            # Parse player slots
            players_str = parts[1]
            p_slots = players_str.split(";")
            for slot_idx, slot_str in enumerate(p_slots):
                if slot_idx >= N_TRACKS or not slot_str or slot_str.startswith("NaN"):
                    continue
                p_items = slot_str.split(",")
                if len(p_items) >= 2 and p_items[0] != "NaN" and p_items[1] != "NaN":
                    player_xy[frame_id, slot_idx, 0] = float(p_items[0]) * pitch_length
                    player_xy[frame_id, slot_idx, 1] = float(p_items[1]) * pitch_width
                if len(p_items) >= 3 and p_items[2] != "NaN":
                    player_speed[frame_id, slot_idx] = float(p_items[2])

    elapsed = time.time() - t0
    print(f"Parsed {n_frames} frames in {elapsed:.1f}s.")

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(
        CACHE_FILE,
        ball_xy=ball_xy,
        ball_speed=ball_speed,
        player_xy=player_xy,
        player_speed=player_speed,
        track_teams=track_teams,
        home_name=home_name,
        away_name=away_name,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
    )
    cache_size_mb = os.path.getsize(CACHE_FILE) / (1024 * 1024)
    print(f"Saved binary cache to {CACHE_FILE} ({cache_size_mb:.1f} MB).")

    return {
        "ball_xy": ball_xy,
        "ball_speed": ball_speed,
        "player_xy": player_xy,
        "player_speed": player_speed,
        "track_teams": track_teams,
        "home_name": home_name,
        "away_name": away_name,
        "pitch": (pitch_length, pitch_width),
    }


def load_cached_atd(data_dir=None):
    """Load cached ATD npz dict."""
    if not os.path.exists(CACHE_FILE):
        return parse_and_cache(data_dir=data_dir)

    data = np.load(CACHE_FILE, allow_pickle=True)
    return {
        "ball_xy": data["ball_xy"],
        "ball_speed": data["ball_speed"],
        "player_xy": data["player_xy"],
        "player_speed": data["player_speed"],
        "track_teams": data["track_teams"],
        "home_name": str(data["home_name"]),
        "away_name": str(data["away_name"]),
        "pitch": (float(data["pitch_length"]), float(data["pitch_width"])),
    }


if __name__ == "__main__":
    force_rebuild = "--force" in sys.argv
    t0 = time.time()
    atd = parse_and_cache(force=force_rebuild)
    ball_vis = np.sum(~np.isnan(atd["ball_xy"][:, 0]))
    print(f"Verification: ball visible in {ball_vis}/{len(atd['ball_xy'])} frames ({100*ball_vis/len(atd['ball_xy']):.1f}%)")
    print(f"Total time: {time.time() - t0:.2f}s")
