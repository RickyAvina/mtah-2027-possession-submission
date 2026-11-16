"""Frame features for the mixture-of-experts HMM.

Works on any dict shaped like solution.parse_atd.load_cached_atd() -- the competition ATD or the
public Metrica games -- and uses tracking only (no Smart Tagging, no video), so a model trained
on external games transfers without touching anything match-specific.

Three things v4 does not do:
  * short ball gaps (<= MAX_INTERP frames) are filled by linear interpolation between the last
    and next detections, and the model is told which frames are interpolated;
  * features look both ways in time (the submission is offline), e.g. "who is next to touch the
    ball" and the mean distance margin over the next second;
  * everything is expressed relative to a team, so the same features with H and A swapped
    describe the mirrored situation -- training uses that to enforce H/A symmetry.
"""

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - pure-python fallback is slow but correct
    def njit(*a, **k):
        def wrap(f):
            return f
        return wrap(a[0]) if a and callable(a[0]) else wrap

FPS = 25
MAX_INTERP = 50          # 2 s
TOUCH_ACC = 10.0         # m/s^2 impulse
TOUCH_DIST = 2.0         # m


# -- helpers ---------------------------------------------------------------------------------

@njit(cache=True)
def _gap_lengths(vis):
    n = len(vis)
    since = np.zeros(n, np.int32)
    until = np.zeros(n, np.int32)
    total = np.zeros(n, np.int32)
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


def _interp_ball(ball, vis, total_gap):
    b = ball.copy()
    idx = np.arange(len(b))
    ok = vis
    fill = (~vis) & (total_gap <= MAX_INTERP)
    if ok.sum() > 1:
        for d in range(2):
            b[fill, d] = np.interp(idx[fill], idx[ok], ball[ok, d])
    # leading/trailing gaps that np.interp clamps are not real interpolation
    first, last = np.flatnonzero(ok)[[0, -1]] if ok.any() else (0, -1)
    fill[:first] = False
    fill[last + 1:] = False
    b[~(vis | fill)] = np.nan
    return b, fill


@njit(cache=True)
def _decay(x, gamma, reverse):
    n = len(x)
    out = np.zeros(n, np.float32)
    acc = 0.0
    if reverse:
        for i in range(n - 1, -1, -1):
            acc = acc * gamma + x[i]
            out[i] = acc
    else:
        for i in range(n):
            acc = acc * gamma + x[i]
            out[i] = acc
    return out


@njit(cache=True)
def _last_event(flag_h, flag_a, reverse):
    """Team (+1 home, -1 away, 0 none) and frames since/until the last/next touch."""
    n = len(flag_h)
    team = np.zeros(n, np.float32)
    dist = np.full(n, 9999.0, np.float32)
    cur, last = 0.0, -1
    rng = range(n - 1, -1, -1) if reverse else range(n)
    for i in rng:
        if flag_h[i] and not flag_a[i]:
            cur, last = 1.0, i
        elif flag_a[i] and not flag_h[i]:
            cur, last = -1.0, i
        team[i] = cur
        if last >= 0:
            dist[i] = abs(i - last)
    return team, dist


@njit(cache=True)
def _run_length(flag):
    n = len(flag)
    out = np.zeros(n, np.int32)
    c = 0
    for i in range(n):
        c = c + 1 if flag[i] else 0
        out[i] = c
    return out


def _window_mean(x, lo, hi):
    """Mean of x over [t+lo, t+hi) ignoring NaN (lo/hi may be negative)."""
    n = len(x)
    v = np.nan_to_num(x, nan=0.0).astype(np.float64)
    ok = (~np.isnan(x)).astype(np.float64)
    cv = np.concatenate(([0.0], np.cumsum(v)))
    co = np.concatenate(([0.0], np.cumsum(ok)))
    t = np.arange(n)
    a = np.clip(t + lo, 0, n)
    b = np.clip(t + hi, 0, n)
    s = cv[b] - cv[a]
    c = co[b] - co[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return out.astype(np.float32)


def _window_max(x, lo, hi):
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(x)
    w = hi - lo
    pad = np.concatenate([np.zeros(max(0, -lo)), x.astype(np.float64),
                          np.zeros(max(0, hi))])  # inputs are >= 0 flags
    off = max(0, -lo) + lo
    sw = sliding_window_view(pad, w)
    return sw[off:off + n].max(axis=1).astype(np.float32)


# -- main ------------------------------------------------------------------------------------

def v4_dead_score(atd):
    """v4's rule-based dead-ball score (out-of-bounds state machine, collective speed collapse,
    missing-and-slow), with no Smart Tagging and no match-specific frames, so it can be computed
    on any match and learned from on the public games."""
    from solution.features import compute_features
    from solution.v4_generic import dead_ball_scores
    return dead_ball_scores(compute_features(atd, events=[]))


def compute(atd):
    """Return (X, names, aux). X is (N, F) float32 with NaN replaced; aux holds raw signals."""
    ball = atd["ball_xy"].astype(np.float32)
    pxy = atd["player_xy"].astype(np.float32)
    pspd = atd["player_speed"].astype(np.float32)
    teams = np.asarray(atd["track_teams"])
    L, W = atd["pitch"]
    n = len(ball)

    vis = ~np.isnan(ball[:, 0])
    since, until, total = _gap_lengths(vis)
    bi, interp = _interp_ball(ball, vis, total)
    have = ~np.isnan(bi[:, 0])

    # player-ball distances
    diff = pxy - bi[:, None, :]
    d = np.sqrt((diff ** 2).sum(-1))
    d[np.isnan(d)] = 200.0
    hm, am = teams == 0, teams == 1
    dh_sorted = np.sort(d[:, hm], axis=1)
    da_sorted = np.sort(d[:, am], axis=1)
    dH, dA = np.minimum(dh_sorted[:, 0], 40), np.minimum(da_sorted[:, 0], 40)
    dH2, dA2 = np.minimum(dh_sorted[:, 1], 40), np.minimum(da_sorted[:, 1], 40)
    nH3, nA3 = (d[:, hm] <= 3).sum(1), (d[:, am] <= 3).sum(1)
    nH8, nA8 = (d[:, hm] <= 8).sum(1), (d[:, am] <= 8).sum(1)
    for arr in (dH, dA, dH2, dA2):
        arr[~have] = np.nan
    for arr in (nH3, nA3, nH8, nA8):
        arr[~have] = 0
    closest = np.fmin(dH, dA)
    margin = np.clip(dA - dH, -15, 15)

    # ball kinematics on the interpolated track
    v = np.full_like(bi, np.nan)
    v[1:-1] = (bi[2:] - bi[:-2]) * (FPS / 2)
    bspd = np.linalg.norm(v, axis=1)
    acc = np.full(n, np.nan, np.float32)
    acc[1:-1] = np.linalg.norm((v[2:] - v[:-2]) * (FPS / 2), axis=1)
    # direction change (deg) between successive velocity vectors
    with np.errstate(invalid="ignore"):
        cosang = (v[2:] * v[:-2]).sum(1) / (np.linalg.norm(v[2:], axis=1) *
                                           np.linalg.norm(v[:-2], axis=1) + 1e-6)
    turn = np.full(n, np.nan, np.float32)
    turn[1:-1] = np.degrees(np.arccos(np.clip(cosang, -1, 1)))

    touch = vis & (np.nan_to_num(acc) > TOUCH_ACC) & (np.nan_to_num(closest, nan=99) <= TOUCH_DIST)
    tH = touch & (dH < dA)
    tA = touch & (dA < dH)
    # "control": ball slow and someone right on it -- a softer cue than a touch impulse
    ctrl = have & (np.nan_to_num(closest, nan=99) <= 1.5) & (np.nan_to_num(bspd, nan=99) < 8)
    cH = ctrl & (dH < dA - 0.5)
    cA = ctrl & (dA < dH - 0.5)

    evH = (tH | cH).astype(np.float32)
    evA = (tA | cA).astype(np.float32)
    momH_p, momA_p = _decay(evH, 0.97, False), _decay(evA, 0.97, False)
    momH_f, momA_f = _decay(evH, 0.97, True), _decay(evA, 0.97, True)
    last_team, since_touch = _last_event(tH | cH, tA | cA, False)
    next_team, until_touch = _last_event(tH | cH, tA | cA, True)

    # pitch geometry
    x, y = bi[:, 0], bi[:, 1]
    inside = have & (x >= 0) & (x <= L) & (y >= 0) & (y <= W)
    oob = have & ~inside
    dx_out = np.maximum(0, np.maximum(-x, x - L))
    dy_out = np.maximum(0, np.maximum(-y, y - W))
    d_out = np.where(oob, np.hypot(dx_out, dy_out), 0.0)
    d_line = np.where(inside, np.minimum.reduce([x, L - x, y, W - y]), 0.0)
    d_goalline = np.minimum(np.abs(x), np.abs(L - x))
    d_touchline = np.minimum(np.abs(y), np.abs(W - y))
    d_center = np.hypot(x - L / 2, y - W / 2)
    near_corner = (d_goalline < 2) & (d_touchline < 2)

    # collective motion
    with np.errstate(all="ignore"):
        mspd = np.nanmean(pspd, axis=1)
        slow_frac = np.nanmean((pspd < 1.0).astype(np.float32) + 0 * pspd, axis=1)
    mspd = np.nan_to_num(mspd, nan=0.0)
    slow_frac = np.nan_to_num(slow_frac, nan=1.0)
    n_tracked = (~np.isnan(pxy[:, :, 0])).sum(1)

    stationary = have & (np.nan_to_num(bspd, nan=99) < 0.7)
    stat_run = _run_length(stationary)
    # restart kick: ball (nearly) still for >= 0.4 s within the last 0.5 s, then struck
    still = vis & (np.nan_to_num(bspd, nan=99) < 1.5)
    still_run = _run_length(still).astype(np.float32)
    recent_still = _window_max(still_run, -12, 1)
    kick = vis & (np.nan_to_num(bspd) > 4.0) & (recent_still >= 10)
    restart_near = _window_max(kick.astype(np.float32), -12, 26) > 0
    ball_seen = _window_mean(vis.astype(np.float32), -25, 0) > 0.8   # watched in the last 1 s
    oob_f = oob.astype(np.float32)

    feats = {}
    def put(name, val):
        feats[name] = np.asarray(val, dtype=np.float32)

    put("vis", vis)
    put("interp", interp)
    put("have", have)
    put("gap_log", np.log1p(total))
    put("since_vis", np.log1p(since))
    put("until_vis", np.log1p(until))
    put("dH", dH); put("dA", dA); put("dH2", dH2); put("dA2", dA2)
    put("closest", closest)
    put("margin", margin)
    put("margin_w", margin / (1 + (np.nan_to_num(closest, nan=10) / 3) ** 2))
    put("nH3", nH3); put("nA3", nA3); put("nH8", nH8); put("nA8", nA8)
    put("bspd", np.clip(bspd, 0, 40)); put("acc", np.clip(np.log1p(acc), 0, 8))
    put("turn", turn)
    put("touch_H", tH); put("touch_A", tA)
    put("ctrl_H", cH); put("ctrl_A", cA)
    put("momH_p", momH_p); put("momA_p", momA_p)
    put("momH_f", momH_f); put("momA_f", momA_f)
    put("mom_p", momH_p - momA_p); put("mom_f", momH_f - momA_f)
    put("last_team", last_team); put("next_team", next_team)
    put("since_touch", np.log1p(since_touch)); put("until_touch", np.log1p(until_touch))
    put("oob", oob); put("d_out", np.clip(d_out, 0, 20)); put("d_line", np.clip(d_line, 0, 34))
    put("d_center", d_center); put("near_corner", near_corner)
    # penalty area / six-yard box of whichever goal is nearer (goal kicks are taken from inside,
    # with the opponents made to wait outside the penalty area)
    y_off = np.abs(y - W / 2)
    put("in_box", have & (d_goalline <= 16.5) & (y_off <= 20.16))
    put("in_six", have & (d_goalline <= 5.5) & (y_off <= 9.16))
    put("mspd", mspd); put("slow_frac", slow_frac); put("n_tracked", n_tracked)
    put("stat_log", np.log1p(stat_run))
    put("restart_kick", kick)
    put("restart_near", restart_near)
    put("ball_seen", ball_seen)
    put("contested", have & (np.nan_to_num(dH, nan=99) <= 2.2) & (np.nan_to_num(dA, nan=99) <= 2.2))
    put("in_flight", have & (np.nan_to_num(bspd) > 10) & (np.nan_to_num(closest, nan=0) > 2))
    v4d = v4_dead_score(atd)
    put("v4_dead", np.clip(v4d, -5, 30) / 10.0)
    put("v4_dead_state", v4d >= 4.0)

    # bidirectional context windows (frames)
    for tag, lo, hi in (("p1", -25, 0), ("f1", 1, 26), ("p3", -75, -25), ("f3", 26, 76)):
        put(f"margin_w_{tag}", _window_mean(feats["margin_w"], lo, hi))
        put(f"mspd_{tag}", _window_mean(mspd, lo, hi))
        put(f"vis_{tag}", _window_mean(vis.astype(np.float32), lo, hi))
        put(f"bspd_{tag}", _window_mean(feats["bspd"], lo, hi))
    for tag, lo, hi in (("p2", -50, 0), ("f2", 1, 51), ("p6", -150, 0), ("f6", 1, 151)):
        put(f"oob_max_{tag}", _window_max(oob_f, lo, hi))
    for tag, lo, hi in (("p1", -25, 0), ("f1", 1, 26), ("p3", -75, -25), ("f3", 26, 76)):
        put(f"v4_dead_{tag}", _window_mean(feats["v4_dead"], lo, hi))
    put("mspd_p6", _window_mean(mspd, -150, 0))
    put("mspd_f6", _window_mean(mspd, 1, 151))

    names = list(feats)
    X = np.stack([feats[k] for k in names], 1)
    nan_mask = np.isnan(X)
    # NaN -> per-feature neutral value; "have" / "vis" already tell the model it is missing
    fill = {"dH": 40, "dA": 40, "dH2": 40, "dA2": 40, "closest": 40, "turn": 0, "bspd": 0,
            "acc": 0, "margin": 0, "margin_w": 0}
    for j, k in enumerate(names):
        X[nan_mask[:, j], j] = fill.get(k, 0.0)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    aux = {"vis": vis, "have": have, "oob": oob, "bi": bi}
    return X, names, aux


# Pairs of columns that swap when the home/away roles are swapped; columns listed in NEGATE
# change sign. Everything else is team-neutral.
SWAP_PAIRS = [("dH", "dA"), ("dH2", "dA2"), ("nH3", "nA3"), ("nH8", "nA8"),
              ("touch_H", "touch_A"), ("ctrl_H", "ctrl_A"), ("momH_p", "momA_p"),
              ("momH_f", "momA_f")]
NEGATE_PREFIX = ("margin", "mom_p", "mom_f", "last_team", "next_team")


def mirror(X, names):
    """Same frames with home and away exchanged."""
    Xm = X.copy()
    ix = {k: i for i, k in enumerate(names)}
    for a, b in SWAP_PAIRS:
        Xm[:, ix[a]], Xm[:, ix[b]] = X[:, ix[b]], X[:, ix[a]]
    for k, i in ix.items():
        if k.startswith(NEGATE_PREFIX):
            Xm[:, i] = -X[:, i]
    return Xm
