"""Mixture-of-experts emissions + duration-aware HMM decoding.

Emission model
--------------
    p(s | x_t) = sum_k  g_k(z_t) * softmax(W_k x_t)_s          s in {H, A, D}

The gate g = softmax(V z_t) sees only a handful of interpretable scenario cues z (ball tracked?,
interpolated?, contested?, in flight?, near/over a line?, players walking?, ball stationary?),
so each expert ends up owning a kind of situation -- open play, aerial / 50-50 duels, ball in
flight, ball missing, set-piece set-up -- while every expert sees the full feature vector x.
Trained by maximum likelihood with L-BFGS (exact gradient), on H/A-mirrored copies so the model
is symmetric in the two teams.

Decoder
-------
Each label is a chain of m sub-states (H: m_live, A: m_live, D: m_dead), so a spell lasts at
least m frames and its length is negative-binomial rather than geometric. That is a cheap
hidden semi-Markov model: possessions shorter than ~1 s become expensive instead of impossible,
replacing v4's hard 3 s post-processing. Emissions enter as tempered scaled likelihoods,
alpha * (log p(s|x) - log prior(s)), because consecutive frames are far from independent.
"""

import json

import numpy as np
from scipy.optimize import minimize

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*a, **k):
        def wrap(f):
            return f
        return wrap(a[0]) if a and callable(a[0]) else wrap

GATE_FEATURES = ["have", "interp", "contested", "in_flight", "oob_max_p2", "oob_max_f2",
                 "closest_lt1_5", "slow", "stat_log", "gap_log", "near_line", "near_center",
                 "v4_dead_state"]


def gate_inputs(X, names):
    ix = {k: i for i, k in enumerate(names)}
    cols = []
    for k in GATE_FEATURES:
        if k == "closest_lt1_5":
            cols.append((X[:, ix["closest"]] < 1.5) & (X[:, ix["have"]] > 0))
        elif k == "slow":
            cols.append(X[:, ix["mspd_p1"]] < 1.3)
        elif k == "near_line":
            cols.append((X[:, ix["d_line"]] < 2.0) & (X[:, ix["have"]] > 0))
        elif k == "near_center":
            cols.append((X[:, ix["d_center"]] < 3.0) & (X[:, ix["have"]] > 0))
        else:
            cols.append(X[:, ix[k]])
    return np.stack(cols, 1).astype(np.float64)


def _softmax(a):
    a = a - a.max(axis=-1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=-1, keepdims=True)


class MoE:
    def __init__(self, n_experts=6, l2=1e-3, seed=0):
        self.K, self.l2, self.seed = n_experts, l2, seed

    # -- parameter packing --
    def _unpack(self, w):
        D, G, K = self.D, self.G, self.K
        n_e = K * (D + 1) * 3
        We = w[:n_e].reshape(K, D + 1, 3)
        Wg = w[n_e:].reshape(G + 1, K)
        return We, Wg

    def _prep(self, X, Z):
        Xs = (X - self.mu) / self.sd
        Zs = (Z - self.zmu) / self.zsd
        Xs = np.hstack([Xs, np.ones((len(Xs), 1))])
        Zs = np.hstack([Zs, np.ones((len(Zs), 1))])
        return Xs, Zs

    def _forward(self, w, Xs, Zs):
        We, Wg = self._unpack(w)
        g = _softmax(Zs @ Wg)                              # (N, K)
        logits = Xs @ We.transpose(1, 0, 2).reshape(Xs.shape[1], -1)
        P = _softmax(logits.reshape(len(Xs), self.K, 3))   # (N, K, 3)
        return g, P

    def fit(self, X, Z, y, maxiter=400, verbose=True):
        self.D, self.G = X.shape[1], Z.shape[1]
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.zmu, self.zsd = Z.mean(0), Z.std(0) + 1e-6
        Xs, Zs = self._prep(X, Z)
        Y = np.eye(3)[y]
        N = len(y)
        rng = np.random.default_rng(self.seed)
        w0 = np.concatenate([rng.normal(0, 0.01, self.K * (self.D + 1) * 3),
                             rng.normal(0, 0.5, (self.G + 1) * self.K)])
        it = [0]

        def f(w):
            We, Wg = self._unpack(w)
            g, P = self._forward(w, Xs, Zs)
            py = (P * Y[:, None, :]).sum(-1)               # (N, K)
            mix = (g * py).sum(1) + 1e-12
            loss = -np.log(mix).mean()
            r = g * py / mix[:, None]                      # responsibilities
            # expert grads: -1/N sum_n r_nk x_n (y_n - P_nk)
            R = (r[:, :, None] * (P - Y[:, None, :])).reshape(N, -1)
            dE = (Xs.T @ R).reshape(self.D + 1, self.K, 3).transpose(1, 0, 2) / N
            dG = Zs.T @ (g - r) / N
            reg_e = We.copy(); reg_e[:, -1, :] = 0         # no penalty on biases
            loss += 0.5 * self.l2 * (reg_e ** 2).sum() + 0.5 * self.l2 * (Wg[:-1] ** 2).sum()
            dE += self.l2 * reg_e
            dGr = self.l2 * Wg.copy(); dGr[-1] = 0
            grad = np.concatenate([dE.ravel(), (dG + dGr).ravel()])
            it[0] += 1
            if verbose and it[0] % 50 == 0:
                print(f"    iter {it[0]:4d} nll={loss:.4f}")
            return loss, grad

        res = minimize(f, w0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
        self.w = res.x
        return self

    def predict_proba(self, X, Z, return_gate=False):
        Xs, Zs = self._prep(X, Z)
        out_p, out_g = [], []
        for i in range(0, len(Xs), 50000):
            g, P = self._forward(self.w, Xs[i:i + 50000], Zs[i:i + 50000])
            out_p.append((g[:, :, None] * P).sum(1))
            out_g.append(g)
        p = np.concatenate(out_p)
        return (p, np.concatenate(out_g)) if return_gate else p

    # -- persistence --
    def to_dict(self):
        return {"K": self.K, "l2": self.l2, "D": self.D, "G": self.G,
                "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                "zmu": self.zmu.tolist(), "zsd": self.zsd.tolist(), "w": self.w.tolist()}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["K"], d["l2"])
        m.D, m.G = d["D"], d["G"]
        for k in ("mu", "sd", "zmu", "zsd", "w"):
            setattr(m, k, np.array(d[k]))
        return m


# -- decoder ---------------------------------------------------------------------------------

@njit(cache=True)
def _viterbi_chains(le, m_live, m_dead, ls_live, ls_dead, l_sw, l_ld, l_dl, dl_extra):
    """Viterbi over chains of sub-states. le: (N, 3) log emission scores for H, A, D.

    Sub-states 0..m_live-1 = H, next m_live = A, next m_dead = D. Inside a chain the only move is
    forward (prob 1) except the last sub-state, which loops (ls_*) or exits to the first
    sub-state of another label (l_sw for H<->A, l_ld for live->D, l_dl for D->H and D->A).
    dl_extra[t] is added to l_dl at frame t (restart gating).
    """
    n = le.shape[0]
    S = 2 * m_live + m_dead
    lab = np.empty(S, np.int64)
    for s in range(S):
        lab[s] = 0 if s < m_live else (1 if s < 2 * m_live else 2)
    first = np.array([0, m_live, 2 * m_live])
    last = np.array([m_live - 1, 2 * m_live - 1, S - 1])
    NEG = -1e18
    dp = np.full(S, NEG)
    for c in range(3):
        dp[first[c]] = le[0, c]
    bp = np.zeros((n, S), np.int32)
    new = np.empty(S)
    for t in range(1, n):
        for s in range(S):
            new[s] = NEG
        # forward moves inside chains
        for c in range(3):
            for s in range(first[c] + 1, last[c] + 1):
                new[s] = dp[s - 1]
                bp[t, s] = s - 1
            # self loop on last sub-state
            stay = ls_live if c < 2 else ls_dead
            v = dp[last[c]] + stay
            if v > new[last[c]]:
                new[last[c]] = v
                bp[t, last[c]] = last[c]
        # exits into first sub-states
        for c_to in range(3):
            best, arg = NEG, -1
            for c_from in range(3):
                if c_from == c_to:
                    continue
                if c_from < 2 and c_to < 2:
                    w = l_sw
                elif c_from < 2:
                    w = l_ld
                else:
                    w = l_dl + dl_extra[t]
                v = dp[last[c_from]] + w
                if v > best:
                    best, arg = v, last[c_from]
            f = first[c_to]
            if f == last[c_to]:  # chain of length 1: compare with self loop already set
                if best > new[f]:
                    new[f] = best
                    bp[t, f] = arg
            elif best > new[f]:
                new[f] = best
                bp[t, f] = arg
        for s in range(S):
            dp[s] = new[s] + le[t, lab[s]]
    path = np.empty(n, np.int64)
    s = 0
    bestv = NEG
    for k in range(S):
        if dp[k] > bestv:
            bestv, s = dp[k], k
    for t in range(n - 1, -1, -1):
        path[t] = lab[s]
        s = bp[t, s]
    return path


DEFAULT_DECODE = {"alpha": 0.15, "m_live": 25, "m_dead": 25, "stay_live": -0.002,
                  "stay_dead": -0.002, "switch": -9.0, "live_dead": -9.0, "dead_live": -8.0,
                  "gate_pen": 0.0}


def restart_gate(X, names):
    """True where play may not restart cheaply: the ball is visible nearby but there is no
    restart kick (ball still, then struck) around this frame. Where the ball is not visible the
    gate is open, so a stoppage can always end even when the restart is not seen."""
    ix = {k: i for i, k in enumerate(names)}
    return (X[:, ix["ball_seen"]] > 0) & (X[:, ix["restart_near"]] <= 0)


FORCE_NONE, FORCE_DEAD, FORCE_LIVE = -1, 2, 5


def decode(proba, prior, runs, n, params=None, force=None, gate=None):
    """Label every frame; frames outside `runs` come back as 3 (X).

    force: optional (n,) int array of hard constraints -- FORCE_DEAD frames must be D,
    FORCE_LIVE frames must be H or A (which one is still decided by the model).
    """
    p = {**DEFAULT_DECODE, **(params or {})}
    proba = np.nan_to_num(proba, nan=1.0 / 3, posinf=1.0, neginf=1e-6)
    le = p["alpha"] * (np.log(np.clip(proba, 1e-6, 1)) - np.log(prior)[None, :])
    if force is not None:
        le = le.copy()
        le[force == FORCE_DEAD, :2] -= 1e6
        le[force == FORCE_LIVE, 2] -= 1e6
    out = np.full(n, 3, np.int64)
    for a, b in runs:
        extra = np.zeros(b - a) if gate is None else -float(p.get("gate_pen", 0.0)) * gate[a:b]
        out[a:b] = _viterbi_chains(le[a:b].astype(np.float64), int(p["m_live"]), int(p["m_dead"]),
                                   p["stay_live"], p["stay_dead"], p["switch"], p["live_dead"],
                                   p["dead_live"], extra.astype(np.float64))
    return out


def save(path, moe, prior, names, decode_params, extra=None):
    json.dump({"moe": moe.to_dict(), "prior": list(map(float, prior)), "features": names,
               "decode": decode_params, **(extra or {})}, open(path, "w"))


def load(path):
    d = json.load(open(path))
    return MoE.from_dict(d["moe"]), np.array(d["prior"]), d["features"], d["decode"], d
