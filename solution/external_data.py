"""Public Metrica Sports sample matches as training / validation data.

The competition terms forbid sending the Barca-Athletic assets to a hosted LLM, so the MoE-HMM
is developed and trained on Metrica's three public sample matches
(https://github.com/metrica-sports/sample-data) and only *run* on the competition ATD locally.

Every loader returns the same dict that solution.parse_atd.load_cached_atd() returns, so the
feature code is shared:

    ball_xy      (N, 2)   metres, origin at a pitch corner, NaN when the ball is not tracked
    ball_speed   (N,)     m/s (derived here; the samples carry no speed channel)
    player_xy    (N, P, 2) metres, NaN when the slot is empty
    player_speed (N, P)   m/s
    track_teams  (P,)     0 = home, 1 = away
    pitch        (105.0, 68.0)

plus, for the external games only:

    labels       (N,)     int8 ground truth: 0 = H, 1 = A, 2 = D, -1 = not graded
    runs         [(start, end), ...]  graded frame ranges, end exclusive (one per period)
    events       pandas DataFrame of the event log (team, type, subtype, start, end)

The H/A/D ground truth is derived mechanically from the event log, following CHALLENGE.md:
possession changes only on a *controlled* action by the other team, stays with the last team
through loose / contested phases, and the ball is dead from the moment it leaves play (or a
whistle) until the restart.
"""

import json
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

EXT_DIR = os.environ.get(
    "MTAH_EXT_DIR", os.path.join(ROOT, "ext_data", "metrica-sample-data", "data"))
PITCH = (105.0, 68.0)
FPS = 25
H, A, D, UNGRADED = 0, 1, 2, -1
GAME_IDS = (1, 2, 3)


# -- tracking --------------------------------------------------------------------------------

def _speeds(xy):
    """Speed (m/s) from positions via central differences, lightly smoothed."""
    v = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    d = (xy[2:] - xy[:-2]) * (FPS / 2.0)
    v[1:-1] = np.linalg.norm(d, axis=-1)
    # 5-frame running mean ignoring NaNs, along time
    k = 5
    vv = np.nan_to_num(v, nan=0.0)
    ok = (~np.isnan(v)).astype(np.float32)
    kern = np.ones(k, dtype=np.float32)
    if v.ndim == 1:
        num = np.convolve(vv, kern, mode="same")
        den = np.convolve(ok, kern, mode="same")
    else:
        num = np.apply_along_axis(lambda c: np.convolve(c, kern, mode="same"), 0, vv)
        den = np.apply_along_axis(lambda c: np.convolve(c, kern, mode="same"), 0, ok)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[ok == 0] = np.nan
    return out.astype(np.float32)


def _load_csv_team(path):
    raw = pd.read_csv(path, skiprows=2)
    cols = list(raw.columns)
    players = [c for c in cols if c.startswith("Player")]
    xy = np.stack([raw[[p, cols[cols.index(p) + 1]]].to_numpy(np.float32) for p in players], 1)
    ball = raw[["Ball", cols[cols.index("Ball") + 1]]].to_numpy(np.float32)
    return raw["Period"].to_numpy(), raw["Frame"].to_numpy(), xy, ball


def _load_csv_game(g):
    base = os.path.join(EXT_DIR, f"Sample_Game_{g}", f"Sample_Game_{g}_RawTrackingData_")
    per, frm, hxy, ball = _load_csv_team(base + "Home_Team.csv")
    _, frm2, axy, _ = _load_csv_team(base + "Away_Team.csv")
    assert np.array_equal(frm, frm2)
    xy = np.concatenate([hxy, axy], axis=1)
    teams = np.array([H] * hxy.shape[1] + [A] * axy.shape[1], dtype=np.int8)
    return per, frm, xy, ball, teams


def _load_epts_game(g):
    folder = os.path.join(EXT_DIR, f"Sample_Game_{g}")
    meta = ET.parse(os.path.join(folder, f"Sample_Game_{g}_metadata.xml")).getroot()
    team_of = {p.get("id"): p.get("teamId") for p in meta.iter("Player")}
    home_id = next(meta.iter("Score")).get("idLocalTeam")
    chan_player = {c.get("id"): c.get("playerId") for c in meta.iter("PlayerChannel")}
    pids = sorted(team_of, key=lambda p: (team_of[p] != home_id, p))
    col = {p: i for i, p in enumerate(pids)}
    teams = np.array([H if team_of[p] == home_id else A for p in pids], dtype=np.int8)

    specs = []
    for spec in meta.iter("DataFormatSpecification"):
        order, seen = [], set()
        for ref in spec.iter("PlayerChannelRef"):
            pid = chan_player[ref.get("playerChannelId")]
            if pid not in seen:
                seen.add(pid)
                order.append(col[pid])
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), order))

    lines = open(os.path.join(folder, f"Sample_Game_{g}_tracking.txt")).read().split("\n")
    lines = [l for l in lines if l]
    n = len(lines)
    frames = np.empty(n, dtype=np.int64)
    xy = np.full((n, len(pids), 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    for i, line in enumerate(lines):
        f, pl, b = line.split(":")
        f = int(f)
        frames[i] = f
        order = next(o for s, e, o in specs if s <= f <= e)
        vals = np.array(re.split("[;,]", pl), dtype=np.float32).reshape(-1, 2)
        xy[i, order] = vals
        bx, by = b.split(",")[:2]
        ball[i] = (float(bx), float(by))

    events = json.load(open(os.path.join(folder, f"Sample_Game_{g}_events.json")))["data"]
    per_starts = {}
    for e in events:
        p = e["period"]
        per_starts[p] = min(per_starts.get(p, 10**9), e["start"]["frame"])
    # periods from events: frames after the 2nd-half kickoff belong to period 2
    p2 = per_starts.get(2, 10**9)
    per = np.where(frames >= p2, 2, 1)
    return per, frames, xy, ball, teams


# -- events ----------------------------------------------------------------------------------

def _events_csv(g):
    e = pd.read_csv(os.path.join(EXT_DIR, f"Sample_Game_{g}", f"Sample_Game_{g}_RawEventsData.csv"))
    return pd.DataFrame({
        "team": e["Team"].map({"Home": H, "Away": A}).astype(int),
        "type": e["Type"].astype(str),
        "subtype": e["Subtype"].fillna("").astype(str),
        "period": e["Period"].astype(int),
        "start": e["Start Frame"].astype(int),
        "end": e["End Frame"].astype(int),
        "sx": e["Start X"].astype(float) * PITCH[0], "sy": e["Start Y"].astype(float) * PITCH[1],
        "ex": e["End X"].astype(float) * PITCH[0], "ey": e["End Y"].astype(float) * PITCH[1],
    })


def _events_json(g):
    folder = os.path.join(EXT_DIR, f"Sample_Game_{g}")
    meta = ET.parse(os.path.join(folder, f"Sample_Game_{g}_metadata.xml")).getroot()
    home_id = next(meta.iter("Score")).get("idLocalTeam")
    rows = []
    for e in json.load(open(os.path.join(folder, f"Sample_Game_{g}_events.json")))["data"]:
        st = e.get("subtypes")
        if isinstance(st, list):
            sub = "-".join(s["name"] for s in st)
        elif isinstance(st, dict):
            sub = st["name"]
        else:
            sub = ""
        rows.append({
            "team": H if e["team"]["id"] == home_id else A,
            "type": e["type"]["name"], "subtype": sub, "period": e["period"],
            "start": e["start"]["frame"], "end": e["end"]["frame"],
            "sx": (e["start"]["x"] if e["start"]["x"] is not None else np.nan) * PITCH[0],
            "sy": (e["start"]["y"] if e["start"]["y"] is not None else np.nan) * PITCH[1],
            "ex": (e["end"]["x"] if e["end"]["x"] is not None else np.nan) * PITCH[0],
            "ey": (e["end"]["y"] if e["end"]["y"] is not None else np.nan) * PITCH[1],
        })
    return pd.DataFrame(rows)


# -- ground truth ----------------------------------------------------------------------------

# Actions that show the acting team in controlled possession at their start frame.
CONTROL = {"PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE", "BALL LOST"}
# BALL LOST subtypes that are whistles rather than turnovers in play.
WHISTLE_LOST = ("OFFSIDE", "HAND BALL", "REFEREE HIT", "END HALF")


def derive_labels(ev, n_index, frame_to_idx, period_of_idx):
    """Per-frame H/A/D from the event log. See module docstring for the rules."""
    ev = ev.sort_values(["start", "end"], kind="stable").reset_index(drop=True)
    labels = np.full(n_index, UNGRADED, dtype=np.int8)

    def idx(f):
        return int(np.clip(np.searchsorted(frame_to_idx, f), 0, n_index - 1))

    runs = []
    for p in sorted(ev["period"].unique()):
        e = ev[ev["period"] == p].reset_index(drop=True)
        kick = e[(e["type"] == "SET PIECE") & e["subtype"].str.contains("KICK OFF")]
        p_start = int(kick["start"].min()) if len(kick) else int(e["start"].min())
        p_end = int(max(e["end"].max(), e["start"].max()))
        # (state, team) change points in frame units
        points = []  # (frame, label)
        dead = False
        for _, r in e.iterrows():
            t, sub, s, en = r["type"], r["subtype"], int(r["start"]), int(r["end"])
            en = max(en, s)
            if t in CONTROL:
                if t == "SET PIECE" or dead or not points:
                    dead = False
                points.append((s, int(r["team"])))
                if t == "BALL LOST" and any(w in sub for w in WHISTLE_LOST):
                    points.append((en, D)); dead = True
                if t == "SHOT" and "GOAL" in sub and "SAVED" not in sub:
                    points.append((en, D)); dead = True
                if t == "SHOT" and sub.endswith("OUT"):
                    points.append((en, D)); dead = True
            elif t == "BALL OUT":
                points.append((en, D)); dead = True
            elif t == "FAULT RECEIVED":
                points.append((s, D)); dead = True
            elif t == "CHALLENGE" and "FAULT" in sub and "ADVANTAGE" not in sub:
                points.append((en, D)); dead = True
        points.sort(key=lambda x: x[0])
        i0, i1 = idx(p_start), idx(p_end) + 1
        cur = points[0][1] if points else D
        lab = np.full(i1 - i0, cur, dtype=np.int8)
        for f, l in points:
            k = idx(f) - i0
            if 0 <= k < len(lab):
                lab[k:] = l
        labels[i0:i1] = lab
        runs.append((i0, i1))
    return labels, runs


# -- public API ------------------------------------------------------------------------------

def load_game(g, cache=True):
    cdir = os.path.join(ROOT, "cache")
    cpath = os.path.join(cdir, f"metrica_game{g}.npz")
    if cache and os.path.exists(cpath):
        z = np.load(cpath, allow_pickle=True)
        ev = _events_csv(g) if g in (1, 2) else _events_json(g)
        return {
            "ball_xy": z["ball_xy"], "ball_speed": z["ball_speed"], "player_xy": z["player_xy"],
            "player_speed": z["player_speed"], "track_teams": z["track_teams"],
            "home_name": "Home", "away_name": "Away", "pitch": PITCH,
            "labels": z["labels"], "runs": [tuple(r) for r in z["runs"]], "events": ev,
            "game": g,
        }

    if g in (1, 2):
        per, frames, xy, ball, teams = _load_csv_game(g)
        ev = _events_csv(g)
    else:
        per, frames, xy, ball, teams = _load_epts_game(g)
        ev = _events_json(g)

    scale = np.array(PITCH, dtype=np.float32)
    player_xy = xy * scale
    ball_xy = ball * scale
    labels, runs = derive_labels(ev, len(frames), frames, per)

    out = {
        "ball_xy": ball_xy.astype(np.float32),
        "ball_speed": _speeds(ball_xy),
        "player_xy": player_xy.astype(np.float32),
        "player_speed": _speeds(player_xy),
        "track_teams": teams,
        "home_name": "Home", "away_name": "Away", "pitch": PITCH,
        "labels": labels, "runs": runs, "events": ev, "game": g,
    }
    if cache:
        os.makedirs(cdir, exist_ok=True)
        np.savez_compressed(cpath, **{k: out[k] for k in (
            "ball_xy", "ball_speed", "player_xy", "player_speed", "track_teams", "labels")},
            runs=np.array(runs))
    return out


# Domain randomisation ranges. The public games track the ball ~97% of live play and almost
# never while it is dead. tools/feed_stats.py on the competition feed (public window) gave:
#   P(visible | live) = 0.935, P(visible | dead) = 0.664, P(out of bounds | dead, visible) = 0.107
# while the whole graded match is only 59% visible -- so the rest of the match loses the ball far
# more often than the public window, and the live-play drop range stays wide. Dead-ball time is
# made mostly visible: the ball rolls on briefly, is carried by the restart taker, then waits at
# the restart spot. Each training copy draws its own setting, so "ball missing" can never be
# used as a proxy for "dead ball" (nor "ball visible" for "live").
AUG_DEFAULT = {
    "live_drop": (0.0, 0.45),       # fraction of live frames with the ball removed
    "live_gap_s": (0.6, 6.0),       # mean length of a live-play gap
    "dead_show": (0.75, 1.0) ,      # fraction of dead segments that get a synthetic ball
    "dead_cover": (0.4, 1.0),       # fraction of such a segment the ball is visible for
    "drift_s": (0.3, 1.5),          # ball keeps rolling this long after play stops
    "spot_s": (1.0, 15.0),          # ball sits at the restart spot this long before restart
    "knock": (0.3, 0.9),            # chance the pre-restart phase is the ball knocked about
                                    # between nearby players rather than carried by the taker
    "player_drop": (0.0, 0.08),     # random player-slot dropout
}


def _restart_spot(ev, f_end):
    """Where play restarts: start xy of the first event at/after frame f_end (event frames)."""
    e = ev[(ev["start"] >= f_end - 2) & np.isfinite(ev["sx"])]
    if not len(e):
        return None
    r = e.iloc[0]
    return float(r["sx"]), float(r["sy"])


def restart_type(ev, f_end):
    """'goal kick', 'throw in', 'corner', 'free kick', 'kick off' or 'other' for the restart
    that ends a stoppage at frame f_end."""
    r = ev[(ev["start"] >= f_end - 2) & (ev["start"] <= f_end + 5)]
    sub = " ".join(r["subtype"].astype(str)).upper()
    for key, name in (("GOAL KICK", "goal kick"), ("THROW IN", "throw in"), ("CORNER", "corner"),
                      ("FREE KICK", "free kick"), ("KICK OFF", "kick off")):
        if key in sub:
            return name
    return "other"


def _knock_about(b, pxy, c0, c1, taker, spot, rng):
    """Ball passed / thrown between nearby players at low speed during a stoppage, held a
    moment by each, ending with the restart taker at the spot -- what the competition feed shows
    while the ball is being brought back, and what live play looks like to a naive model."""
    t = c0
    pos = b[c0 - 1] if c0 > 0 and not np.isnan(b[c0 - 1, 0]) else pxy[c0, taker]
    if np.isnan(pos[0]):
        return
    while t < c1:
        d = np.linalg.norm(pxy[t] - pos, axis=1)
        near = np.flatnonzero(d < 25)
        last_leg = (c1 - t) < 3 * FPS
        tgt = taker if (last_leg or not len(near)) else int(rng.choice(near))
        dest = spot if last_leg else pxy[t, tgt]
        if np.isnan(dest[0]):
            dest = spot
        speed = rng.uniform(3, 12)
        L = int(np.clip(np.linalg.norm(dest - pos) / speed * FPS, 2, c1 - t))
        frac = np.linspace(0, 1, L + 1)[1:, None]
        b[t:t + L] = pos + (dest - pos) * (1 - (1 - frac) ** 1.6)   # decelerating roll
        t += L
        pos = np.array(dest, dtype=np.float32)
        hold = min(int(rng.uniform(0.5, 3.0) * FPS), c1 - t)
        if hold > 0:
            tr = pxy[t:t + hold, tgt] if not last_leg else np.repeat(pos[None], hold, 0)
            tr = np.where(np.isnan(tr), pos, tr) + rng.normal(0, 0.4, size=2)  # fixed offset
            b[t:t + hold] = tr
            pos = b[t + hold - 1].copy()
            t += hold


def augment(game, rng, cfg=None):
    """One randomised copy of a game whose ball track looks like the competition feed."""
    cfg = {**AUG_DEFAULT, **(cfg or {})}
    u = lambda k: rng.uniform(*cfg[k])
    lab = game["labels"]
    b = game["ball_xy"].copy()
    n = len(b)
    vis = ~np.isnan(b[:, 0])
    live = (lab == H) | (lab == A)
    pxy0 = game["player_xy"]

    # 1. drop the ball in live play, biased toward fast / airborne moments
    drop, mean_gap = u("live_drop"), u("live_gap_s")
    spd = np.nan_to_num(game["ball_speed"], nan=0.0)
    w = (1.0 + np.clip(spd / 8.0, 0, 3)) * live
    w = w / w.sum()
    target = int(drop * live.sum())
    removed = 0
    keep = vis.copy()
    while removed < target:
        s = int(rng.choice(n, p=w))
        L = max(3, int(rng.exponential(mean_gap * FPS)))
        e = min(n, s + L)
        removed += int((keep[s:e] & live[s:e]).sum())
        keep[s:e] &= ~live[s:e]
    b[~keep] = np.nan

    # 2. synthetic ball during most dead segments: drift -> (hole) -> carried -> at the spot
    show = u("dead_show")
    ev = game["events"]
    for s0, e0 in game["runs"]:
        x = lab[s0:e0]
        br = np.flatnonzero(x[1:] != x[:-1]) + 1
        bd = np.concatenate(([0], br, [len(x)])) + s0
        for k in range(len(bd) - 1):
            a, z = int(bd[k]), int(bd[k + 1])
            seg = z - a
            if lab[a] != D or seg < 5:
                continue
            # goal kicks: the keeper fetches the ball and holds / walks with it in view for most
            # of the stoppage, then places and plays it (seen on the competition feed)
            goal_kick = restart_type(ev, z + 1) == "goal kick"
            if not goal_kick and rng.random() > show:
                continue
            cover = int((rng.uniform(0.7, 1.0) if goal_kick else u("dead_cover")) * seg)
            # (a) drift: continue last observed velocity, decelerating
            nd = 0
            last = a - 1
            while last > a - 50 and last > 0 and np.isnan(game["ball_xy"][last, 0]):
                last -= 1
            if last > 2 and not np.isnan(game["ball_xy"][last - 2, 0]):
                p0 = game["ball_xy"][last]
                v0 = (game["ball_xy"][last] - game["ball_xy"][last - 2]) * (FPS / 2)
                nd = min(int(u("drift_s") * FPS), seg, cover)
                t = np.arange(1, nd + 1) / FPS
                decay = np.exp(-1.5 * t)[:, None]
                b[a:a + nd] = p0 + v0 * ((1 - decay) / 1.5)
            spot = _restart_spot(ev, z + 1)
            if spot is None:
                continue
            spot = np.array(spot, dtype=np.float32)
            rest = max(0, cover - nd)
            ns = min(int((rng.uniform(0.5, 4.0) if goal_kick else u("spot_s")) * FPS), rest)
            nc = rest - ns
            # (c) at the spot for the last ns frames
            if ns > 0:
                b[z - ns:z] = spot + rng.normal(0, 0.02, size=(ns, 2))
            # (b) carried by the restart taker (player nearest the spot at the restart)
            if nc > 0:
                c1 = z - ns
                c0 = max(a + nd, c1 - nc)
                dz = np.linalg.norm(pxy0[min(z, n - 1)] - spot, axis=1)
                if np.all(np.isnan(dz)):
                    continue
                taker = int(np.nanargmin(dz))
                if not goal_kick and rng.random() < u("knock"):
                    _knock_about(b, pxy0, c0, c1, taker, spot, rng)
                else:
                    off = rng.normal(0, 0.35 if goal_kick else 0.6, size=2)
                    tr = pxy0[c0:c1, taker] + off
                    ok = ~np.isnan(tr[:, 0])
                    b[c0:c1][ok] = tr[ok]

    # 3. player-slot dropout
    pxy = game["player_xy"].copy()
    pd_ = u("player_drop")
    if pd_ > 0:
        mask = rng.random(pxy.shape[:2]) < pd_
        # make dropouts contiguous-ish: dilate along time
        mask = mask | np.roll(mask, 1, 0) | np.roll(mask, 2, 0)
        pxy[mask] = np.nan
    g2 = dict(game)
    g2["ball_xy"] = b.astype(np.float32)
    g2["ball_speed"] = _speeds(b.astype(np.float32))
    g2["player_xy"] = pxy
    return g2


if __name__ == "__main__":
    for g in GAME_IDS:
        gm = load_game(g)
        lab = gm["labels"]
        graded = lab != UNGRADED
        vis = ~np.isnan(gm["ball_xy"][:, 0])
        # spells: consecutive H/A runs, D breaks
        n_sp, durs = 0, []
        for s, e in gm["runs"]:
            x = lab[s:e]
            br = np.flatnonzero(x[1:] != x[:-1]) + 1
            bd = np.concatenate(([0], br, [len(x)]))
            for i in range(len(bd) - 1):
                if x[bd[i]] in (H, A):
                    n_sp += 1
                    durs.append((bd[i + 1] - bd[i]) / FPS)
        print(f"game {g}: frames={len(lab)} graded={graded.sum()} runs={gm['runs']} "
              f"ball_vis={vis[graded].mean():.1%} H={np.mean(lab[graded]==H):.2f} "
              f"A={np.mean(lab[graded]==A):.2f} D={np.mean(lab[graded]==D):.2f} "
              f"spells={n_sp} mean={np.mean(durs):.1f}s median={np.median(durs):.1f}s")
