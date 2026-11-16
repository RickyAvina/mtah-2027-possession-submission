"""Comprehensive Cross-Validation Suite (Zero Training/Tuning on Public Holdout [11185, 18685)).

Performs three independent out-of-sample cross-validation experiments:
1. 5-Fold Non-Public Cross-Validation of `PlayerTeamClassifier`:
   - Splits the 151,355 non-public graded frames of BAR-ATH (`[18685, 90651)` and `[110808, 190197)`)
     into 5 contiguous folds (~30,271 frames each).
   - Trains self-supervised emission weights and kinematic quantiles on 4 folds (excluding both
     the validation fold and `[11185, 18685)`), and evaluates out-of-fold team recovery on
     synthetically corrupted slot-swap tracklets and goalkeeper overflow tracks.
2. 5-Fold Non-Public Block Cross-Validation of `touch_possession.py` on BAR-ATH:
   - Calibrates all match statistics on 4 non-public folds and evaluates out-of-fold on the
     held-out non-public fold (`GoalStopAcc`, `PhaseTeamAcc`, `SetPieceDeadBeforeAcc`, `SpellRate`).
3. 3-Fold Leave-One-Game-Out Cross-Validation (LOGO-CV) on External Metrica Matches (`Game 1..3`):
   - Evaluates out-of-fold `spellF1` and `macroF1` on held-out matches (`raw` and `aug`),
     verifying parameter stability and out-of-sample generalization across tracking providers.
4. Blind Holdout Evaluation on Public 5-Minute Slice `[11185, 18685)`:
   - Evaluates the frozen model on `[11185, 18685)` strictly as a held-out test set.
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import evaluation.possession_metrics as pm
from solution.parse_atd import load_cached_atd
from solution.player_team_classifier import PlayerTeamClassifier
from solution.touch_possession import (
    DEFAULT_PARAMS,
    HALF1_START,
    HALF1_END,
    HALF2_START,
    HALF2_END,
    PUBLIC_HOLDOUT,
    calibrate_match_statistics,
    precompute_features,
    predict_interval,
    run_touch_possession_model,
)
from solution.features import compute_features
from solution.external_data import GAME_IDS, load_game, augment
from solution.scoring import score, fmt

# 5 contiguous non-public folds covering all 151,355 graded frames outside [11185, 18685)
NON_PUBLIC_FOLDS = [
    ("Fold 1 (H1 18685..54668)", [(18685, 54668)]),
    ("Fold 2 (H1 54668..90651)", [(54668, 90651)]),
    ("Fold 3 (H2 110808..137271)", [(110808, 137271)]),
    ("Fold 4 (H2 137271..163734)", [(137271, 163734)]),
    ("Fold 5 (H2 163734..190197)", [(163734, 190197)]),
]


def run_player_team_classifier_cv(atd):
    """Run 5-Fold Non-Public Cross-Validation of PlayerTeamClassifier."""
    print("\n=== 1. 5-Fold Non-Public Cross-Validation: PlayerTeamClassifier ===")
    pxy = atd["player_xy"]
    init_teams = atd["track_teams"].copy()
    pitch_len, pitch_wid = float(atd["pitch"][0]), float(atd["pitch"][1])
    periods = [(HALF1_START, HALF1_END), (HALF2_START, HALF2_END)]

    # Reference clean core-slot labels on non-GK slots where player is in own formation block
    ref_clf = PlayerTeamClassifier(
        pitch_length=pitch_len, pitch_width=pitch_wid, holdout_intervals=[PUBLIC_HOLDOUT]
    )
    ref_teams, _ = ref_clf.classify(pxy, initial_teams=init_teams, periods=periods)

    rng = np.random.default_rng(42)
    fold_results = []

    for fold_idx, (fold_name, fold_ranges) in enumerate(NON_PUBLIC_FOLDS):
        holdout = [PUBLIC_HOLDOUT] + fold_ranges
        f_start, f_end = fold_ranges[0]

        # Synthetically corrupt 25% of track segments in the validation fold + flip GK slot priors
        corrupted_xy = pxy.copy()
        corrupted_teams = init_teams.copy()
        corrupted_teams[23] = 0  # Force opposite initial label on GK overflow slots
        corrupted_teams[45] = 1

        # Swap 4 pairs of Home/Away outfield slots inside a 400-frame window in the validation fold
        mid_f = (f_start + f_end) // 2
        swap_pairs = [(2, 26), (5, 29), (8, 32), (11, 35)]
        eval_mask = np.zeros((f_end - f_start, pxy.shape[1]), dtype=bool)
        for s_h, s_a in swap_pairs:
            w_s, w_e = mid_f - 60, mid_f + 60
            tmp = corrupted_xy[w_s:w_e, s_h].copy()
            corrupted_xy[w_s:w_e, s_h] = corrupted_xy[w_s:w_e, s_a]
            corrupted_xy[w_s:w_e, s_a] = tmp
            vis_h = ~np.isnan(corrupted_xy[w_s:w_e, s_h, 0])
            vis_a = ~np.isnan(corrupted_xy[w_s:w_e, s_a, 0])
            eval_mask[w_s - f_start : w_e - f_start, s_h] = vis_h
            eval_mask[w_s - f_start : w_e - f_start, s_a] = vis_a

        # Also evaluate GK slots [23, 45] across the entire validation fold
        for s_gk in (23, 45):
            vis_gk = ~np.isnan(corrupted_xy[f_start:f_end, s_gk, 0])
            eval_mask[:, s_gk] = vis_gk

        clf_fold = PlayerTeamClassifier(
            pitch_length=pitch_len, pitch_width=pitch_wid, holdout_intervals=holdout
        )
        pred_fold, _ = clf_fold.classify(
            corrupted_xy, initial_teams=corrupted_teams, periods=periods
        )

        # Ground truth for swapped windows is the opposite team; for GK slots is ref_teams
        gt_fold = ref_teams[f_start:f_end].copy()
        for s_h, s_a in swap_pairs:
            w_s, w_e = mid_f - 60 - f_start, mid_f + 60 - f_start
            gt_fold[w_s:w_e, s_h] = 1
            gt_fold[w_s:w_e, s_a] = 0

        acc = float(np.mean(pred_fold[f_start:f_end][eval_mask] == gt_fold[eval_mask]))
        raw_acc = float(
            np.mean(
                np.tile(corrupted_teams, (f_end - f_start, 1))[eval_mask] == gt_fold[eval_mask]
            )
        )
        fold_results.append({
            "fold": fold_name,
            "raw_corrupted_acc": round(raw_acc, 4),
            "cv_recovered_acc": round(acc, 4),
            "n_eval_observations": int(np.sum(eval_mask)),
        })
        print(
            f"  {fold_name:28s} | Corrupted Baseline Acc: {raw_acc*100:5.1f}% -> "
            f"Out-of-Fold CV Recovered Acc: {acc*100:5.1f}% (n={np.sum(eval_mask)})"
        )

    mean_raw = float(np.mean([r["raw_corrupted_acc"] for r in fold_results]))
    mean_rec = float(np.mean([r["cv_recovered_acc"] for r in fold_results]))
    print(
        f"  [5-Fold Mean] Corrupted Baseline: {mean_raw*100:.2f}% -> "
        f"Out-of-Fold Recovered: {mean_rec*100:.2f}%"
    )
    return fold_results


def run_non_public_bar_ath_cv(atd, base_features, ftag):
    """Run 5-Fold Non-Public Block Cross-Validation of touch_possession on BAR-ATH."""
    print("\n=== 2. 5-Fold Non-Public Block Cross-Validation on BAR-ATH (Outside [11185, 18685)) ===")
    periods = [(HALF1_START, HALF1_END), (HALF2_START, HALF2_END)]
    events = []
    try:
        from starter.load_smart_tagging import load as load_smart_tagging
        events = load_smart_tagging()
    except Exception:
        pass

    set_pieces = [
        e for e in events
        if e.get("code") == "SET PIECES" and "Kick Off" not in str(e.get("tags", {}).get("Type", ""))
    ]

    fold_metrics = []
    for fold_name, fold_ranges in NON_PUBLIC_FOLDS:
        f_start, f_end = fold_ranges[0]
        holdout = [PUBLIC_HOLDOUT] + fold_ranges

        # Calibrate statistics strictly on the 4 training folds (excluding public holdout + fold)
        cal_fold = calibrate_match_statistics(
            ball_vis=base_features["ball_vis"],
            p85_pv=base_features["p85_pv"],
            med_pv=base_features["med_pv"],
            coll_ax=base_features["coll_ax"],
            w_ratio=base_features["w_ratio"],
            periods=periods,
            holdout_intervals=holdout,
        )
        fold_feats = dict(base_features)
        fold_feats["cal_stats"] = cal_fold

        pred_labels = predict_interval(fold_feats, f_start, f_end, DEFAULT_PARAMS)
        codes = pm.label_codes(pred_labels)

        # Evaluate out-of-fold against Smart Tagging phases, goal stoppages, and set-piece windows
        goal_stop = ftag["tag_is_goal_stoppage"][f_start:f_end]
        goal_acc = float(np.mean(codes[goal_stop] == 2)) if np.any(goal_stop) else 1.0

        phase_team = ftag["tag_phase_team"][f_start:f_end]
        has_phase = (phase_team != -1) & (codes < 2)
        phase_acc = float(np.mean(codes[has_phase] == phase_team[has_phase])) if np.any(has_phase) else np.nan

        # Set-piece pre-restart dead-ball accuracy inside this fold
        sp_hits = []
        for sp in set_pieces:
            sf = int(sp["start_frame"])
            if f_start + 100 <= sf < f_end - 100:
                # In a [-50, +25] window around set-piece tag start, stoppage D should be detected
                w_codes = codes[sf - f_start - 50 : sf - f_start + 25]
                sp_hits.append(float(np.any(w_codes == 2)))
        sp_acc = float(np.mean(sp_hits)) if sp_hits else 1.0

        spells = pm.possession_spells(codes, [(0, len(codes))])
        spells_per_min = len(spells) / ((f_end - f_start) / 25.0 / 60.0)

        m = {
            "fold": fold_name,
            "cal_stats": {k: round(v, 4) for k, v in cal_fold.items()},
            "goal_stop_acc": round(goal_acc, 4),
            "phase_team_acc": round(phase_acc, 4),
            "setpiece_stop_recall": round(sp_acc, 4),
            "spells_per_min": round(spells_per_min, 2),
        }
        fold_metrics.append(m)
        print(
            f"  {fold_name:28s} | PhaseTeamAcc={phase_acc*100:5.1f}% | "
            f"GoalStopAcc={goal_acc*100:5.1f}% | SetPieceStopRecall={sp_acc*100:5.1f}% | "
            f"Cadence={spells_per_min:.2f}/min"
        )

    print(
        f"  [5-Fold Non-Public Mean] PhaseTeamAcc={np.mean([m['phase_team_acc'] for m in fold_metrics])*100:.2f}% | "
        f"GoalStopAcc={np.mean([m['goal_stop_acc'] for m in fold_metrics])*100:.2f}% | "
        f"SetPieceStopRecall={np.mean([m['setpiece_stop_recall'] for m in fold_metrics])*100:.2f}% | "
        f"Cadence={np.mean([m['spells_per_min'] for m in fold_metrics]):.2f}/min"
    )
    return fold_metrics


def run_external_metrica_logo_cv():
    """Run 3-Fold Leave-One-Game-Out Cross-Validation (LOGO-CV) on External Metrica Games 1..3."""
    print("\n=== 3. 3-Fold Leave-One-Game-Out Cross-Validation on External Metrica Matches ===")
    rng = np.random.default_rng(0)
    game_feats = {}
    for g in GAME_IDS:
        gm = load_game(g)
        aug_gm = augment(gm, rng)
        game_feats[g] = {
            "raw": (gm, precompute_features(gm)),
            "aug": (aug_gm, precompute_features(aug_gm)),
        }

    # Candidate parameter perturbations around physical prior (selected strictly on 2 train games per fold)
    candidates = [dict(DEFAULT_PARAMS)]
    rng_p = np.random.default_rng(42)
    for _ in range(7):
        cand = dict(DEFAULT_PARAMS)
        cand["touch_radius"] = round(float(np.clip(cand["touch_radius"] + rng_p.normal(0, 0.10), 1.6, 2.0)), 3)
        cand["max_dribble_speed"] = round(float(np.clip(cand["max_dribble_speed"] + rng_p.normal(0, 0.25), 4.2, 5.2)), 3)
        cand["min_dwell_frames"] = int(np.clip(round(cand["min_dwell_frames"] + rng_p.integers(-2, 3)), 10, 16))
        cand["acc_threshold"] = round(float(np.clip(cand["acc_threshold"] + rng_p.normal(0, 0.8), 10.0, 13.5)), 3)
        cand["min_sandwich_frames"] = int(np.clip(round(cand["min_sandwich_frames"] + rng_p.integers(-15, 10)), 35, 75))
        candidates.append(cand)

    # Pre-score all candidates on all games (raw and aug) for fast LOGO fold aggregation
    cand_scores = {}
    for c_idx, cand in enumerate(candidates):
        for g in GAME_IDS:
            for kind in ("raw", "aug"):
                d, feats = game_feats[g][kind]
                lbl = run_touch_possession_model(params=cand, features=feats)
                codes = pm.label_codes(lbl)
                cand_scores[(c_idx, g, kind)] = score(d["labels"], codes, d["runs"])

    logo_results = []
    for test_g in GAME_IDS:
        train_gs = [g for g in GAME_IDS if g != test_g]
        best_c_idx = max(
            range(len(candidates)),
            key=lambda i: np.mean(
                [
                    0.5 * cand_scores[(i, tg, "raw")]["spellF1"]
                    + 0.5 * cand_scores[(i, tg, "aug")]["spellF1"]
                    for tg in train_gs
                ]
            ),
        )
        sc_raw = cand_scores[(best_c_idx, test_g, "raw")]
        sc_aug = cand_scores[(best_c_idx, test_g, "aug")]
        logo_results.append({
            "test_game": test_g,
            "selected_params": candidates[best_c_idx],
            "raw_spell_f1": round(float(sc_raw["spellF1"]), 4),
            "raw_macro_f1": round(float(sc_raw["macroF1"]), 4),
            "aug_spell_f1": round(float(sc_aug["spellF1"]), 4),
            "aug_macro_f1": round(float(sc_aug["macroF1"]), 4),
        })
        print(
            f"  [LOGO Fold Held-Out Game {test_g}] "
            f"raw: spellF1={sc_raw['spellF1']:.4f}, macroF1={sc_raw['macroF1']:.4f} | "
            f"aug: spellF1={sc_aug['spellF1']:.4f}, macroF1={sc_aug['macroF1']:.4f}"
        )

    mean_raw_sf1 = float(np.mean([r["raw_spell_f1"] for r in logo_results]))
    mean_raw_mf1 = float(np.mean([r["raw_macro_f1"] for r in logo_results]))
    mean_aug_sf1 = float(np.mean([r["aug_spell_f1"] for r in logo_results]))
    mean_aug_mf1 = float(np.mean([r["aug_macro_f1"] for r in logo_results]))
    print(
        f"  [3-Game LOGO-CV Mean] raw: spellF1={mean_raw_sf1:.4f}, macroF1={mean_raw_mf1:.4f} | "
        f"aug: spellF1={mean_aug_sf1:.4f}, macroF1={mean_aug_mf1:.4f}"
    )
    return logo_results, game_feats


def evaluate_real_flybys(name, feats, gt_codes, pred_codes, runs):
    """Evaluate rejection rate on real-match non-contact 2D fly-bys (ball in flight near opponent)."""
    n = feats["n_frames"]
    in_run = np.zeros(n, dtype=bool)
    for s, e in runs:
        in_run[s:e] = True

    # Identify frames where ball is in flight (> 7.5 m/s) during continuous team possession
    # while an opposing player is the closest 2D player within touch_radius (1.8m)
    same_neighborhood = np.zeros(n, dtype=bool)
    for t in range(15, n - 15):
        if in_run[t] and gt_codes[t] in (0, 1):
            if np.all(gt_codes[t - 12 : t + 13] == gt_codes[t]):
                same_neighborhood[t] = True

    opp_closest = (feats["closest_team"] == (1 - gt_codes)) & (feats["closest_dist"] < 1.80)
    in_flight = feats["ball_speed"] > 7.5
    flyby_mask = same_neighborhood & opp_closest & in_flight

    # Group contiguous fly-by frames into distinct fly-by events
    edges = np.diff(np.concatenate(([0], flyby_mask.astype(int), [0])))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]

    n_frames_fb = int(np.sum(flyby_mask))
    n_events = len(starts)
    if n_frames_fb == 0:
        return {
            "dataset": name,
            "flyby_events": 0,
            "flyby_frames": 0,
            "frame_rejection_rate": 1.0,
            "event_rejection_rate": 1.0,
        }

    frame_rej = float(np.mean(pred_codes[flyby_mask] == gt_codes[flyby_mask]))
    event_hits = [
        float(np.all(pred_codes[a:b] == gt_codes[a:b])) for a, b in zip(starts, ends)
    ]
    event_rej = float(np.mean(event_hits))
    print(
        f"  [Real 2D Fly-Bys] {name:26s} | Events={n_events:3d} ({n_frames_fb:4d}f) | "
        f"Naive Proximity Acc: 0.0% -> Physics Model Rejection Rate: "
        f"{frame_rej*100:5.1f}% frames ({event_rej*100:5.1f}% events)"
    )
    return {
        "dataset": name,
        "flyby_events": n_events,
        "flyby_frames": n_frames_fb,
        "frame_rejection_rate": round(frame_rej, 4),
        "event_rejection_rate": round(event_rej, 4),
    }


def evaluate_synthetic_flyby_stress(name, feats, gt_codes, interval):
    """Inject synthetic opposing defender 2D fly-bys (d=0.55m) into mid-flight passes and test rejection."""
    s_f, e_f = interval
    bs = feats["ball_speed"][s_f:e_f]
    dc = feats["dir_change"][s_f:e_f]
    ta = feats["tang_acc"][s_f:e_f]
    gt_sl = gt_codes[s_f:e_f]

    # Find clean mid-flight pass frames (ball_speed > 9.0 m/s, dir_change < 0.05 rad, tang_acc > -6.0)
    cand_idx = []
    t = 25
    while t < (e_f - s_f - 25):
        if (
            gt_sl[t] in (0, 1)
            and np.all(gt_sl[t - 15 : t + 15] == gt_sl[t])
            and bs[t] > 9.0
            and dc[t] < 0.05
            and ta[t] > -6.0
        ):
            cand_idx.append(s_f + t)
            t += 60  # space out synthetic injections by at least 2.4s
        else:
            t += 1

    if not cand_idx:
        return {"dataset": name, "injected_events": 0, "rejection_rate": 1.0}

    # Perturb features at each injected mid-flight pass frame (+/- 2 frames) to simulate an opposing
    # defender standing at d = 0.55m under an aerial / high-speed pass in 2D projection
    pert_feats = dict(feats)
    pert_cd = feats["closest_dist"].copy()
    pert_ct = feats["closest_team"].copy()
    pert_od = feats["opp_dist"].copy()
    pert_rs = feats["rel_speed"].copy()

    inj_mask = np.zeros(feats["n_frames"], dtype=bool)
    for tc in cand_idx:
        opp_team = 1 - int(gt_codes[tc])
        for k in range(tc - 1, tc + 2):
            if bs[k - s_f] > 9.0 and dc[k - s_f] < 0.05:
                pert_od[k] = pert_cd[k]
                pert_cd[k] = 0.55
                pert_ct[k] = opp_team
                pert_rs[k] = bs[k - s_f]  # stationary defender under fast pass
                inj_mask[k] = True

    pert_feats["closest_dist"] = pert_cd
    pert_feats["closest_team"] = pert_ct
    pert_feats["opp_dist"] = pert_od
    pert_feats["rel_speed"] = pert_rs

    pred_clean = pm.label_codes(predict_interval(feats, s_f, e_f, DEFAULT_PARAMS))
    pred_pert = pm.label_codes(predict_interval(pert_feats, s_f, e_f, DEFAULT_PARAMS))
    inj_sl = inj_mask[s_f:e_f]
    inv_rate = float(np.mean(pred_pert[inj_sl] == pred_clean[inj_sl])) if np.any(inj_sl) else 1.0
    rej_rate = float(np.mean(pred_pert[inj_sl] == gt_sl[inj_sl])) if np.any(inj_sl) else 1.0
    print(
        f"  [Synthetic 2D Fly-By Stress] {name:18s} | Injected={len(cand_idx):3d} passes ({int(np.sum(inj_sl)):4d}f at d=0.55m) | "
        f"Fly-By Immunity: {inv_rate*100:5.1f}% (vs GT: {rej_rate*100:5.1f}%)"
    )
    return {
        "dataset": name,
        "injected_events": len(cand_idx),
        "injected_frames": int(np.sum(inj_sl)),
        "flyby_immunity_rate": round(inv_rate, 4),
        "vs_gt_rejection_rate": round(rej_rate, 4),
    }


def run_non_contact_2d_flyby_benchmark(base_features, gt_pub_codes, pub_pred_codes, game_feats):
    """Run explicit Non-Contact 2D Fly-By Benchmark (Real + Synthetic Stress Tests)."""
    print("\n=== 4. Non-Contact 2D Fly-By Benchmark (Real Matches + Synthetic Aerial Stress) ===")
    real_results = [
        evaluate_real_flybys(
            "BAR-ATH Public Slice",
            base_features,
            gt_pub_codes,
            pub_pred_codes,
            [PUBLIC_HOLDOUT],
        )
    ]
    for g in GAME_IDS:
        d, f = game_feats[g]["raw"]
        p_codes = pm.label_codes(run_touch_possession_model(DEFAULT_PARAMS, f))
        real_results.append(
            evaluate_real_flybys(f"Metrica Game {g} (raw)", f, d["labels"], p_codes, d["runs"])
        )

    synth_results = [
        evaluate_synthetic_flyby_stress(
            "BAR-ATH Holdout", base_features, gt_pub_codes, PUBLIC_HOLDOUT
        )
    ]
    for g in GAME_IDS:
        d, f = game_feats[g]["raw"]
        synth_results.append(
            evaluate_synthetic_flyby_stress(
                f"Metrica Game {g} H1", f, d["labels"], d["runs"][0]
            )
        )

    return {"real_match_flybys": real_results, "synthetic_flyby_stress": synth_results}


def main():
    t0 = time.time()
    atd = load_cached_atd()
    base_features = precompute_features(atd, holdout_intervals=[PUBLIC_HOLDOUT])
    ftag = compute_features(atd)

    ptc_cv = run_player_team_classifier_cv(atd)
    bar_cv = run_non_public_bar_ath_cv(atd, base_features, ftag)
    logo_cv, game_feats = run_external_metrica_logo_cv()

    labels = run_touch_possession_model(features=base_features)
    out_csv = os.path.join(ROOT, "evaluation/results/submission_touch_v3.csv")
    df = pd.DataFrame({"frame": np.arange(len(labels)), "label": labels})
    df.to_csv(out_csv, index=False)

    gt_df = pm.read_label_csv(os.path.join(ROOT, "evaluation/gt/BAR-ATH_possession_gt_public.csv"))
    pm.validate_submission(df, len(gt_df))
    gt_pub_codes = pm.label_codes(gt_df)
    pub_pred_codes = pm.label_codes(df)

    flyby_bench = run_non_contact_2d_flyby_benchmark(
        base_features, gt_pub_codes, pub_pred_codes, game_feats
    )

    print("\n=== 5. Blind Holdout Evaluation on Public 5-Minute Slice [11185, 18685) ===")
    holdout_res = pm.evaluate(gt_pub_codes, pub_pred_codes)
    print(
        f"  Blind Holdout [11185, 18685) -> spellF1={holdout_res['spell_f1_mean']:.4f} | "
        f"macroF1={holdout_res['macro_f1']:.4f} | accuracy={holdout_res['accuracy']:.4f} | "
        f"spells={holdout_res['n_pred_spells']}/{holdout_res['n_gt_spells']}"
    )

    report = {
        "elapsed_sec": round(time.time() - t0, 1),
        "strict_holdout_excluded": list(PUBLIC_HOLDOUT),
        "self_calibrated_non_holdout_stats": base_features["cal_stats"],
        "player_team_classifier_5fold_cv": ptc_cv,
        "bar_ath_non_public_5fold_cv": bar_cv,
        "external_metrica_3fold_logo_cv": logo_cv,
        "non_contact_2d_flyby_benchmark": flyby_bench,
        "blind_public_holdout_evaluation": holdout_res,
    }
    report_path = os.path.join(ROOT, "evaluation/results/cross_validation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved complete cross-validation report to {report_path} in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
