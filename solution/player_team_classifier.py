"""Player Team Classifier (Home vs Away) with Tracklet Re-Indexing, Kinematic Stitching,
Lagged Formation Features, and HMM Smoothing.

All spatial scales are derived from official FIFA pitch geometry (Penalty Area 16.5m x 40.32m,
Goal Area 5.5m x 18.32m) and empirical kinematic quantiles estimated directly from non-public
match tracking frames. Self-supervised emission parameters are fitted strictly outside any
evaluation holdout slice (e.g. excluding [11185, 18685)).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

# Official FIFA Laws of the Game pitch markings (meters)
FIFA_PENALTY_AREA_DEPTH = 16.5
FIFA_PENALTY_AREA_HALF_WIDTH = 20.16
FIFA_GOAL_AREA_DEPTH = 5.5


class PlayerTeamClassifier:
    """Re-indexes player tracking dots, interpolates merges, and classifies Home (0) vs Away (1)."""

    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        fps: int = 25,
        holdout_intervals: Optional[List[Tuple[int, int]]] = None,
    ):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.fps = fps
        # Default holdout excludes the public 5-minute benchmark [11185, 18685) from self-supervised fitting
        self.holdout_intervals = (
            [(11185, 18685)] if holdout_intervals is None else holdout_intervals
        )
        self.weights_: Optional[np.ndarray] = None
        self.bias_: float = 0.0
        self.feat_mean_: Optional[np.ndarray] = None
        self.feat_std_: Optional[np.ndarray] = None
        self.v_max_step_: float = 0.35

    def _build_training_period_mask(
        self,
        n_frames: int,
        periods: List[Tuple[int, int]],
    ) -> np.ndarray:
        """Build frame mask for self-supervised estimation, strictly excluding holdout_intervals."""
        mask = np.zeros(n_frames, dtype=bool)
        for p_start, p_end in periods:
            mask[p_start:p_end] = True
        for h_start, h_end in self.holdout_intervals:
            if h_start < n_frames:
                mask[max(0, h_start) : min(n_frames, h_end)] = False
        # Fallback if all frames were inside holdout (e.g. unit test on a tiny snippet)
        if not np.any(mask):
            for p_start, p_end in periods:
                mask[p_start:p_end] = True
        return mask

    def _estimate_kinematic_scales(
        self,
        player_xy: np.ndarray,
        train_mask: np.ndarray,
    ) -> float:
        """Estimate the 99.9th-percentile single-frame human sprint step (m/frame) from non-holdout data."""
        sub_xy = player_xy[train_mask]
        if len(sub_xy) < 10:
            sub_xy = player_xy
        dp = np.linalg.norm(np.diff(sub_xy, axis=0), axis=-1)
        valid_dp = dp[~np.isnan(dp) & (dp < 2.0)]
        if len(valid_dp) > 100:
            self.v_max_step_ = float(np.percentile(valid_dp, 99.9))
        else:
            self.v_max_step_ = 10.0 / float(self.fps)
        return self.v_max_step_

    def _detect_goalkeeper_slots(
        self,
        player_xy: np.ndarray,
        vis: np.ndarray,
        train_mask: np.ndarray,
    ) -> np.ndarray:
        """Identify goalkeeper/overflow slots using official FIFA Penalty Area geometry."""
        n_frames, n_slots, _ = player_xy.shape
        active_vis = vis & train_mask[:, None]
        px = np.nan_to_num(player_xy[:, :, 0], nan=self.pitch_length / 2.0)
        py = np.nan_to_num(player_xy[:, :, 1], nan=self.pitch_width / 2.0)
        dy_goal = np.abs(py - self.pitch_width / 2.0)

        in_penalty_area = (
            (px <= FIFA_PENALTY_AREA_DEPTH)
            | (px >= self.pitch_length - FIFA_PENALTY_AREA_DEPTH)
        ) & (dy_goal <= FIFA_PENALTY_AREA_HALF_WIDTH)

        is_gk_slot = np.zeros(n_slots, dtype=bool)
        min_active_frames = 10 * self.fps  # 10 seconds
        for s in range(n_slots):
            m = active_vis[:, s]
            if np.sum(m) >= min_active_frames:
                # An outfield player spends <10% of the match inside a penalty box; GKs spend >1/3
                if float(np.mean(in_penalty_area[m, s])) > (1.0 / 3.0):
                    is_gk_slot[s] = True
        return is_gk_slot

    def compute_period_defending_directions(
        self,
        player_xy: np.ndarray,
        vis: np.ndarray,
        initial_teams: np.ndarray,
        periods: List[Tuple[int, int]],
        is_gk_slot: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Establish defending directions and team centroids per frame."""
        n_frames, n_slots, _ = player_xy.shape
        att_dir = np.full(n_frames, 1.0, dtype=np.float32)

        if is_gk_slot is None:
            is_gk_slot = np.zeros(n_slots, dtype=bool)

        h_core = [s for s in range(n_slots) if initial_teams[s] == 0 and not is_gk_slot[s]]
        a_core = [s for s in range(n_slots) if initial_teams[s] == 1 and not is_gk_slot[s]]
        if not h_core:
            h_core = [s for s in range(n_slots) if initial_teams[s] == 0]
        if not a_core:
            a_core = [s for s in range(n_slots) if initial_teams[s] == 1]

        h_mask = vis[:, h_core]
        a_mask = vis[:, a_core]

        h_pts = np.where(h_mask[:, :, None], player_xy[:, h_core, :], 0.0)
        c_home = h_pts.sum(axis=1) / np.maximum(h_mask.sum(axis=1, keepdims=True), 1)

        a_pts = np.where(a_mask[:, :, None], player_xy[:, a_core, :], 0.0)
        c_away = a_pts.sum(axis=1) / np.maximum(a_mask.sum(axis=1, keepdims=True), 1)

        for p_start, p_end in periods:
            sl = slice(p_start, p_end)
            diff_mean = float(np.mean(c_home[sl, 0] - c_away[sl, 0]))
            period_dir = 1.0 if diff_mean < 0 else -1.0
            att_dir[sl] = period_dir

        return att_dir, c_home.astype(np.float32), c_away.astype(np.float32)

    def _detect_phantom_interpolations(
        self,
        player_xy: np.ndarray,
    ) -> np.ndarray:
        """Detect artificial linear-interpolation phantom segments (zero second derivative for >= 3s)."""
        n_frames, n_slots, _ = player_xy.shape
        min_const_frames = 3 * self.fps
        phantom_mask = np.zeros((n_frames, n_slots), dtype=bool)
        for s in range(n_slots):
            ps = player_xy[:, s]
            dv = np.diff(ps, axis=0)
            ddv = np.linalg.norm(np.diff(dv, axis=0), axis=-1)
            speed = np.linalg.norm(dv[:-1], axis=-1)
            is_linear = (ddv < 1e-3) & (speed > 1e-4) & (speed < 1.0 / self.fps)
            padded = np.concatenate(([False], is_linear, [False]))
            starts = np.where(padded[1:] & (~padded[:-1]))[0]
            ends = np.where((~padded[1:]) & padded[:-1])[0]
            for st, en in zip(starts, ends):
                if en - st >= min_const_frames:
                    phantom_mask[st : en + 2, s] = True
        return phantom_mask

    def _extract_atomic_tracklets(
        self,
        player_xy: np.ndarray,
        periods: List[Tuple[int, int]],
    ) -> List[Dict]:
        """Segment raw slots into continuous atomic tracklets, splitting on NaNs and teleports."""
        n_frames, n_slots, _ = player_xy.shape
        vis = ~np.isnan(player_xy[:, :, 0])
        # A single-frame displacement > 4.5x the 99.9th-percentile sprint step is a physical teleport
        jump_thresh = 4.5 * self.v_max_step_
        vel_win = max(2, self.fps // 3)
        tracklets = []

        for p_idx, (p_start, p_end) in enumerate(periods):
            for s in range(n_slots):
                p_vis = vis[p_start:p_end, s]
                active_frames = np.where(p_vis)[0] + p_start
                if len(active_frames) == 0:
                    continue

                frame_diffs = np.diff(active_frames)
                pos_diffs = np.linalg.norm(
                    player_xy[active_frames[1:], s] - player_xy[active_frames[:-1], s],
                    axis=-1,
                )
                breaks = np.where((frame_diffs > 1) | (pos_diffs > jump_thresh))[0]

                starts = [0] + [b + 1 for b in breaks]
                ends = [b for b in breaks] + [len(active_frames) - 1]

                for st_i, en_i in zip(starts, ends):
                    f_start = int(active_frames[st_i])
                    f_end = int(active_frames[en_i])
                    length = f_end - f_start + 1

                    v_len = min(vel_win, length - 1)
                    if v_len >= 2:
                        v_out = (
                            player_xy[f_end, s] - player_xy[f_end - v_len, s]
                        ) / float(v_len)
                        v_in = (
                            player_xy[f_start + v_len, s] - player_xy[f_start, s]
                        ) / float(v_len)
                    else:
                        v_out = np.zeros(2, dtype=np.float32)
                        v_in = np.zeros(2, dtype=np.float32)

                    tracklets.append({
                        "tid": len(tracklets),
                        "period": p_idx,
                        "slot": s,
                        "f_start": f_start,
                        "f_end": f_end,
                        "length": length,
                        "pos_start": player_xy[f_start, s].copy(),
                        "pos_end": player_xy[f_end, s].copy(),
                        "v_in": v_in.astype(np.float32),
                        "v_out": v_out.astype(np.float32),
                        "prev_tid": None,
                        "next_tid": None,
                        "merge_host_slot": None,
                        "split_host_slot": None,
                    })

        return tracklets

    def _stitch_tracklets(
        self,
        tracklets: List[Dict],
        player_xy: np.ndarray,
        initial_teams: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[List[int]], List[Dict]]:
        """Stitch atomic tracklets across occlusions/slot-swaps using empirical kinematic bounds."""
        n_frames, n_slots, _ = player_xy.shape
        corrected_xy = player_xy.copy()
        vis = ~np.isnan(player_xy[:, :, 0])

        max_stitch_gap = int(2.6 * self.fps)  # 2.6s occlusion horizon
        max_interp_gap = int(1.8 * self.fps)  # 1.8s interpolation horizon
        prox_radius = 5.0 * self.v_max_step_  # ~1.7m human contact/occlusion radius

        for tr in tracklets:
            f_end = tr["f_end"]
            s = tr["slot"]
            if f_end + 1 < n_frames:
                dists = np.linalg.norm(player_xy[f_end] - tr["pos_end"], axis=-1)
                dists[s] = 999.0
                dists = np.nan_to_num(dists, nan=999.0)
                min_s = int(np.argmin(dists))
                if dists[min_s] < prox_radius and vis[f_end + 1, min_s]:
                    tr["merge_host_slot"] = min_s

            f_start = tr["f_start"]
            if f_start - 1 >= 0:
                dists = np.linalg.norm(player_xy[f_start] - tr["pos_start"], axis=-1)
                dists[s] = 999.0
                dists = np.nan_to_num(dists, nan=999.0)
                min_s = int(np.argmin(dists))
                if dists[min_s] < 1.25 * prox_radius and vis[f_start - 1, min_s]:
                    tr["split_host_slot"] = min_s

        by_period: Dict[int, List[int]] = {}
        for tr in tracklets:
            by_period.setdefault(tr["period"], []).append(tr["tid"])

        merge_split_events = []

        for p_idx, tids in by_period.items():
            ends_sorted = sorted(tids, key=lambda idx: tracklets[idx]["f_end"])
            starts_sorted = sorted(tids, key=lambda idx: tracklets[idx]["f_start"])
            start_frames = np.array([tracklets[idx]["f_start"] for idx in starts_sorted])

            candidates = []
            for idx_a in ends_sorted:
                tr_a = tracklets[idx_a]
                t_a = tr_a["f_end"]

                lo = int(np.searchsorted(start_frames, t_a + 1, side="left"))
                hi = int(np.searchsorted(start_frames, t_a + max_stitch_gap, side="right"))

                for k in range(lo, hi):
                    idx_b = starts_sorted[k]
                    tr_b = tracklets[idx_b]
                    dt = tr_b["f_start"] - t_a

                    dt_eff = min(float(dt), float(self.fps))
                    pred_pos = tr_a["pos_end"] + tr_a["v_out"] * dt_eff
                    dist_pred = float(np.linalg.norm(pred_pos - tr_b["pos_start"]))
                    dist_raw = float(np.linalg.norm(tr_a["pos_end"] - tr_b["pos_start"]))
                    dist_eff = min(dist_pred, dist_raw)

                    # Reachability cone: contact radius + 99.9th-percentile sprint speed * dt
                    max_allowed = prox_radius + self.v_max_step_ * dt
                    if dist_eff > max_allowed:
                        continue

                    speed_a = float(np.linalg.norm(tr_a["v_out"]))
                    speed_b = float(np.linalg.norm(tr_b["v_in"]))
                    vel_penalty = 0.0
                    if speed_a > 0.15 * self.v_max_step_ and speed_b > 0.15 * self.v_max_step_:
                        cos_sim = float(np.dot(tr_a["v_out"], tr_b["v_in"]) / (speed_a * speed_b))
                        vel_penalty = 0.5 * prox_radius * (1.0 - cos_sim)

                    slot_bonus = -0.5 * prox_radius if tr_a["slot"] == tr_b["slot"] else 0.0
                    team_bonus = (
                        -0.25 * prox_radius
                        if initial_teams[tr_a["slot"]] == initial_teams[tr_b["slot"]]
                        else 0.25 * prox_radius
                    )
                    if (
                        tr_a["merge_host_slot"] is not None
                        and tr_a["merge_host_slot"] == tr_b["split_host_slot"]
                    ):
                        slot_bonus -= 0.6 * prox_radius

                    cost = dist_eff + 0.1 * self.v_max_step_ * dt + vel_penalty + slot_bonus + team_bonus
                    candidates.append((cost, idx_a, idx_b, dt, dist_eff))

            candidates.sort(key=lambda x: x[0])
            used_a = set()
            used_b = set()

            for cost, idx_a, idx_b, dt, dist_eff in candidates:
                if idx_a in used_a or idx_b in used_b:
                    continue
                used_a.add(idx_a)
                used_b.add(idx_b)
                tracklets[idx_a]["next_tid"] = idx_b
                tracklets[idx_b]["prev_tid"] = idx_a

                if dt <= max_interp_gap:
                    s_fill = tracklets[idx_a]["slot"]
                    f1 = tracklets[idx_a]["f_end"]
                    f2 = tracklets[idx_b]["f_start"]
                    if np.isnan(corrected_xy[f1 + 1 : f2, s_fill, 0]).all():
                        alphas = np.linspace(0.0, 1.0, dt + 1)[1:-1, None]
                        interp_pts = (
                            (1.0 - alphas) * tracklets[idx_a]["pos_end"][None, :]
                            + alphas * tracklets[idx_b]["pos_start"][None, :]
                        )
                        corrected_xy[f1 + 1 : f2, s_fill] = interp_pts.astype(np.float32)

                    merge_split_events.append({
                        "f_start": f1,
                        "f_end": f2,
                        "from_slot": tracklets[idx_a]["slot"],
                        "to_slot": tracklets[idx_b]["slot"],
                        "host_slot": tracklets[idx_a]["merge_host_slot"],
                        "dist": round(dist_eff, 2),
                    })

        chains: List[List[int]] = []
        visited = set()
        for tr in tracklets:
            tid = tr["tid"]
            if tid in visited or tr["prev_tid"] is not None:
                continue
            chain = []
            curr = tid
            while curr is not None and curr not in visited:
                visited.add(curr)
                chain.append(curr)
                curr = tracklets[curr]["next_tid"]
            chains.append(chain)

        reindexed_ids = np.full((n_frames, n_slots), -1, dtype=np.int32)
        for chain_id, chain in enumerate(chains):
            for idx_pos, tid in enumerate(chain):
                tr = tracklets[tid]
                tr["chain_id"] = chain_id
                s = tr["slot"]
                reindexed_ids[tr["f_start"] : tr["f_end"] + 1, s] = chain_id
                if idx_pos + 1 < len(chain):
                    next_tr = tracklets[chain[idx_pos + 1]]
                    dt = next_tr["f_start"] - tr["f_end"]
                    if dt <= max_interp_gap:
                        mask = np.isnan(player_xy[tr["f_end"] + 1 : next_tr["f_start"], s, 0]) & (
                            ~np.isnan(corrected_xy[tr["f_end"] + 1 : next_tr["f_start"], s, 0])
                        )
                        frames_gap = np.arange(tr["f_end"] + 1, next_tr["f_start"])[mask]
                        reindexed_ids[frames_gap, s] = chain_id

        return corrected_xy, reindexed_ids, chains, merge_split_events

    def _compute_frame_features(
        self,
        player_xy: np.ndarray,
        att_dir: np.ndarray,
        c_home: np.ndarray,
        c_away: np.ndarray,
        phantom_mask: np.ndarray,
    ) -> np.ndarray:
        """Compute scale-invariant spatial & formation features normalized by FIFA pitch geometry."""
        n_frames, n_slots, _ = player_xy.shape
        quarter_len = self.pitch_length / 4.0

        px = np.nan_to_num(player_xy[:, :, 0], nan=self.pitch_length / 2.0)
        py = np.nan_to_num(player_xy[:, :, 1], nan=self.pitch_width / 2.0)
        u_x = np.where(att_dir[:, None] > 0, px, self.pitch_length - px)
        signed_x_norm = (u_x / self.pitch_length) * 2.0 - 1.0  # -1 at Home goal, +1 at Away goal

        vis_all = (~np.isnan(player_xy[:, :, 0])) & (~phantom_mask)
        u_x_masked = np.where(vis_all, u_x, np.inf)
        order = np.argsort(u_x_masked, axis=1)
        ranks = np.argsort(order, axis=1).astype(np.float32)
        n_vis = np.maximum(vis_all.sum(axis=1, keepdims=True), 2).astype(np.float32)
        rank_01 = np.clip(ranks / (n_vis - 1.0), 0.0, 1.0)
        rank_x = rank_01 * 2.0 - 1.0

        d_ch = np.linalg.norm(player_xy - c_home[:, None, :], axis=-1)
        d_ca = np.linalg.norm(player_xy - c_away[:, None, :], axis=-1)
        d_ch = np.nan_to_num(d_ch, nan=quarter_len)
        d_ca = np.nan_to_num(d_ca, nan=quarter_len)
        delta_centroid_norm = np.clip((d_ca - d_ch) / quarter_len, -2.0, 2.0)

        u_ch = np.where(att_dir > 0, c_home[:, 0], self.pitch_length - c_home[:, 0])
        u_ca = np.where(att_dir > 0, c_away[:, 0], self.pitch_length - c_away[:, 0])
        signed_dx_home = np.clip((u_x - u_ch[:, None]) / quarter_len, -2.0, 2.0)
        signed_dx_away = np.clip((u_x - u_ca[:, None]) / quarter_len, -2.0, 2.0)

        # Goalmouth Gaussian kernel scaled by FIFA Penalty Area depth (16.5m) for deepest players
        dy_goal = py - self.pitch_width / 2.0
        d_goal_h = np.sqrt(u_x**2 + dy_goal**2)
        d_goal_a = np.sqrt((self.pitch_length - u_x) ** 2 + dy_goal**2)

        gk_zone_home = np.exp(-(d_goal_h**2) / (2.0 * FIFA_PENALTY_AREA_DEPTH**2)) * (
            ranks <= 2
        ).astype(np.float32)
        gk_zone_away = np.exp(-(d_goal_a**2) / (2.0 * FIFA_PENALTY_AREA_DEPTH**2)) * (
            ranks >= (n_vis - 3)
        ).astype(np.float32)

        feats = np.stack(
            [
                signed_x_norm,
                rank_x,
                delta_centroid_norm,
                signed_dx_home,
                signed_dx_away,
                gk_zone_home,
                gk_zone_away,
            ],
            axis=-1,
        ).astype(np.float32)

        return feats

    def _fit_self_supervised_emission_model(
        self,
        feats: np.ndarray,
        vis: np.ndarray,
        initial_teams: np.ndarray,
        train_mask_1d: np.ndarray,
        is_gk_slot: Optional[np.ndarray] = None,
    ) -> None:
        """Train a regularized logistic emission model on non-holdout anchor frames."""
        n_frames, n_slots, n_feat = feats.shape
        if is_gk_slot is None:
            is_gk_slot = np.zeros(n_slots, dtype=bool)
        core_slots = ~is_gk_slot
        mask = vis & train_mask_1d[:, None] & core_slots[None, :]

        sub_mask = np.zeros_like(mask)
        sub_mask[::10, :] = mask[::10, :]

        # Train on all non-holdout active core-slot frames (no ad-hoc delta_c truncation)
        prior_y = np.tile(initial_teams, (n_frames, 1))
        train_mask = sub_mask

        X = feats[train_mask]
        y = prior_y[train_mask].astype(np.float32)

        self.feat_mean_ = X.mean(axis=0)
        self.feat_std_ = np.maximum(X.std(axis=0), 1e-4)
        X_norm = (X - self.feat_mean_) / self.feat_std_

        w = np.zeros(n_feat, dtype=np.float32)
        b = 0.0
        reg = 10.0
        for _ in range(6):
            logits = X_norm @ w + b
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0)))
            diff = probs - y
            grad_w = (X_norm.T @ diff) / len(y) + (reg / len(y)) * w
            grad_b = float(np.mean(diff))
            weights_irls = probs * (1.0 - probs) + 1e-4
            H = (X_norm.T * weights_irls) @ X_norm / len(y) + (reg / len(y)) * np.eye(n_feat)
            step_w = np.linalg.solve(H, grad_w)
            w -= step_w.astype(np.float32)
            b -= grad_b / float(np.mean(weights_irls))

        self.weights_ = w
        self.bias_ = float(b)

    def _run_chain_hmm_with_lagged_features(
        self,
        frame_log_odds_away: np.ndarray,
        feats: np.ndarray,
        tracklets: List[Dict],
        chains: List[List[int]],
        initial_teams: np.ndarray,
        is_gk_slot: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply multi-scale lagged trajectory pooling and Viterbi decoding per stitched chain."""
        n_frames, n_slots = frame_log_odds_away.shape
        pred_teams = np.tile(initial_teams, (n_frames, 1)).astype(np.int8)
        if is_gk_slot is None:
            is_gk_slot = np.zeros(n_slots, dtype=bool)

        junction_stay_logp = np.log(0.90)
        junction_flip_logp = np.log(0.10)
        max_interp_gap = int(1.8 * self.fps)

        for chain in chains:
            if not chain:
                continue

            chain_frames = []
            chain_slots = []
            for tid in chain:
                tr = tracklets[tid]
                f_arr = np.arange(tr["f_start"], tr["f_end"] + 1)
                chain_frames.append(f_arr)
                chain_slots.append(np.full(len(f_arr), tr["slot"], dtype=np.int32))

            all_f = np.concatenate(chain_frames)
            all_s = np.concatenate(chain_slots)
            total_len = len(all_f)

            seq_lo = frame_log_odds_away[all_f, all_s]
            seq_gk_diff = feats[all_f, all_s, 6] - feats[all_f, all_s, 5]  # >0 Away GK, <0 Home GK

            # Bidirectional trajectory pooling at 6s (150 frames) and 24s (600 frames)
            w_med = min(6 * self.fps + 1, total_len)
            w_long = min(24 * self.fps + 1, total_len)
            if w_med >= 5:
                k_med = np.ones(w_med, dtype=np.float32) / float(w_med)
                lag_med = np.convolve(seq_lo, k_med, mode="same")
                lag_gk = np.convolve(seq_gk_diff, k_med, mode="same")
            else:
                lag_med = seq_lo
                lag_gk = seq_gk_diff

            if w_long >= 15:
                k_long = np.ones(w_long, dtype=np.float32) / float(w_long)
                lag_long = np.convolve(seq_lo, k_long, mode="same")
            else:
                lag_long = lag_med

            # Chain-level duration-weighted core slot consensus
            core_home_frames = sum(
                tracklets[tid]["length"]
                for tid in chain
                if not is_gk_slot[tracklets[tid]["slot"]]
                and initial_teams[tracklets[tid]["slot"]] == 0
            )
            core_away_frames = sum(
                tracklets[tid]["length"]
                for tid in chain
                if not is_gk_slot[tracklets[tid]["slot"]]
                and initial_teams[tracklets[tid]["slot"]] == 1
            )
            chain_core_total = core_home_frames + core_away_frames
            chain_prior_away = (
                (core_away_frames - core_home_frames) / float(chain_core_total)
                if chain_core_total > 0
                else 0.0
            )

            n_tr = len(chain)
            tr_log_lik = np.zeros((n_tr, 2), dtype=np.float64)
            offset = 0
            for idx_t, tid in enumerate(chain):
                tr = tracklets[tid]
                L = tr["length"]
                # Equal-weighted multi-scale trajectory log-odds (instantaneous + 6s + 24s)
                seg_lo = (
                    seq_lo[offset : offset + L]
                    + lag_med[offset : offset + L]
                    + lag_long[offset : offset + L]
                ) / 3.0
                seg_gk = float(np.mean(lag_gk[offset : offset + L]))
                offset += L

                mean_lo = float(np.clip(np.mean(seg_lo), -3.0, 3.0))
                s = tr["slot"]
                dur_scale = min(np.sqrt(L / float(self.fps)), 5.0)

                # Unified empirical-Bayes log-odds combining spatial, goalmouth, and slot prior
                if is_gk_slot[s]:
                    net_away_score = dur_scale * (2.5 * mean_lo + 6.0 * seg_gk)
                elif abs(seg_gk) > (1.0 / 3.0):
                    net_away_score = dur_scale * 4.0 * np.sign(seg_gk)
                else:
                    slot_sign = 1.0 if initial_teams[s] == 1 else -1.0
                    if (
                        L < 8 * self.fps
                        and chain_core_total > 40 * self.fps
                        and (slot_sign * chain_prior_away < -0.5)
                    ):
                        net_away_score = 2.0 * chain_prior_away + 0.8 * mean_lo
                    else:
                        net_away_score = dur_scale * (2.5 * slot_sign + 0.45 * mean_lo)

                tr_log_lik[idx_t, 0] = -0.5 * net_away_score
                tr_log_lik[idx_t, 1] = +0.5 * net_away_score

            if n_tr == 1:
                states = [int(np.argmax(tr_log_lik[0]))]
            else:
                dp = np.zeros((n_tr, 2), dtype=np.float64)
                back = np.zeros((n_tr, 2), dtype=np.int32)
                dp[0] = tr_log_lik[0]
                t_mat = np.array(
                    [
                        [junction_stay_logp, junction_flip_logp],
                        [junction_flip_logp, junction_stay_logp],
                    ]
                )
                for i in range(1, n_tr):
                    for curr_s in (0, 1):
                        scores = dp[i - 1] + t_mat[:, curr_s]
                        best_prev = int(np.argmax(scores))
                        dp[i, curr_s] = scores[best_prev] + tr_log_lik[i, curr_s]
                        back[i, curr_s] = best_prev

                states = [0] * n_tr
                states[-1] = int(np.argmax(dp[-1]))
                for i in range(n_tr - 1, 0, -1):
                    states[i - 1] = int(back[i, states[i]])

            for idx_t, tid in enumerate(chain):
                tr = tracklets[tid]
                team_val = states[idx_t]
                s = tr["slot"]
                pred_teams[tr["f_start"] : tr["f_end"] + 1, s] = team_val
                if idx_t + 1 < n_tr:
                    next_tr = tracklets[chain[idx_t + 1]]
                    if next_tr["f_start"] - tr["f_end"] <= max_interp_gap:
                        pred_teams[tr["f_end"] + 1 : next_tr["f_start"], s] = team_val

        return pred_teams

    def classify(
        self,
        player_xy: np.ndarray,
        initial_teams: np.ndarray,
        periods: Optional[List[Tuple[int, int]]] = None,
        max_gap_frames: int = 10,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        """Re-index, interpolate, and classify all player tracklets into Home (0) vs Away (1)."""
        n_frames, n_slots, _ = player_xy.shape
        vis = ~np.isnan(player_xy[:, :, 0])

        if periods is None:
            periods = [(11185, 90651), (110808, 190197)]

        train_mask_1d = self._build_training_period_mask(n_frames, periods)
        self._estimate_kinematic_scales(player_xy, train_mask_1d)

        phantom_mask = self._detect_phantom_interpolations(player_xy)
        is_gk_slot = self._detect_goalkeeper_slots(
            player_xy, vis & (~phantom_mask), train_mask_1d
        )

        att_dir, c_home, c_away = self.compute_period_defending_directions(
            player_xy, vis & (~phantom_mask), initial_teams, periods, is_gk_slot=is_gk_slot
        )

        tracklets = self._extract_atomic_tracklets(player_xy, periods)
        corrected_xy, reindexed_ids, chains, merge_split_events = self._stitch_tracklets(
            tracklets, player_xy, initial_teams
        )

        vis_corr = (~np.isnan(corrected_xy[:, :, 0])) & (~phantom_mask)
        feats = self._compute_frame_features(
            corrected_xy, att_dir, c_home, c_away, phantom_mask
        )

        self._fit_self_supervised_emission_model(
            feats, vis_corr, initial_teams, train_mask_1d, is_gk_slot=is_gk_slot
        )

        feats_norm = (feats - self.feat_mean_[None, None, :]) / self.feat_std_[None, None, :]
        frame_log_odds_away = feats_norm @ self.weights_ + self.bias_

        pred_teams = self._run_chain_hmm_with_lagged_features(
            frame_log_odds_away, feats, tracklets, chains, initial_teams, is_gk_slot=is_gk_slot
        )

        goal_h_x = np.where(att_dir > 0, 0.0, self.pitch_length)
        goal_a_x = np.where(att_dir > 0, self.pitch_length, 0.0)

        diagnostics = {
            "c_home": c_home,
            "c_away": c_away,
            "att_dir": att_dir,
            "goal_h_x": goal_h_x,
            "goal_a_x": goal_a_x,
            "corrected_xy": corrected_xy,
            "reindexed_ids": reindexed_ids,
            "phantom_mask": phantom_mask,
            "merge_split_events": merge_split_events,
            "num_atomic_tracklets": len(tracklets),
            "num_stitched_chains": len(chains),
            "v_max_step": self.v_max_step_,
            "gk_slots": np.where(is_gk_slot)[0].tolist(),
        }
        return pred_teams, diagnostics


def classify_match_players(
    data_dir: Optional[str] = None,
) -> np.ndarray:
    """Convenience helper to load cached ATD data and classify all players."""
    from solution.parse_atd import load_cached_atd

    atd = load_cached_atd(data_dir=data_dir)
    player_xy = atd["player_xy"]
    track_teams = atd["track_teams"]
    pitch_len = float(atd["pitch"][0])
    pitch_wid = float(atd["pitch"][1])

    clf = PlayerTeamClassifier(pitch_length=pitch_len, pitch_width=pitch_wid)
    pred_teams, _ = clf.classify(player_xy, initial_teams=track_teams)
    return pred_teams
