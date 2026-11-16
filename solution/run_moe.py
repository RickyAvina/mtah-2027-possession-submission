"""Run the MoE-HMM (trained on public Metrica games) on the competition ATD, locally.

    python solution/run_moe.py                       # writes evaluation/results/submission_moe.csv

The decoder settings used are the ones chosen on the external games. Do NOT pick settings by
scoring them on the public window: EVALUATION.md shows that even one baseline knob chosen there
generalises the wrong way to the full match. --tune-decoder exists only as a diagnostic (how much
headroom is in the decoder at all) and writes its CSV under a separate name.

Nothing here sends data anywhere: it reads cache/tactical_atd.npz (python solution/parse_atd.py
builds it), the model weights in solution/moe_model.json, and the public GT already in the repo.

The hand labels overlap the public window, so they are not an independent check either.
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import evaluation.possession_metrics as pm  # noqa: E402
from solution import moe_features as mf  # noqa: E402
from solution import moe_hmm as mh  # noqa: E402
from solution.parse_atd import load_cached_atd  # noqa: E402
from solution.scoring import fmt, score  # noqa: E402

GRADED_RUNS = [(11185, 90652), (110808, 190198)]
LABEL = np.array(["H", "A", "D", "X"], dtype=object)


def gt_codes(name):
    return pm.label_codes(pm.read_label_csv(os.path.join(ROOT, "evaluation", "gt", name)))


def sparse_codes(name, n):
    import pandas as pd
    df = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", name))
    out = np.full(n, pm.X, dtype=np.int64)
    out[df["frame"].to_numpy()] = pm.label_codes(df["label"].to_numpy())
    return out


def match_constraints(atd, n):
    """Hard constraints from the match's own structure (not fitted to any ground truth):
      * dead from the start of the 2nd graded half until the 2nd-half kickoff, and
        live for the first 15 frames after each kickoff (kickoff frames published in
        starter/load_smart_tagging.py, KICKOFF_VIDEO_FRAMES);
      * dead during Smart Tagging goal stoppages (goal -> kick off after goal), as v4 does.
    """
    from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES
    force = np.full(n, mh.FORCE_NONE, dtype=np.int64)
    k1, k2 = KICKOFF_VIDEO_FRAMES["1st Half"], KICKOFF_VIDEO_FRAMES["2nd Half"]
    force[GRADED_RUNS[1][0]:k2] = mh.FORCE_DEAD
    n_goal = 0
    try:
        from solution.features import compute_features
        goal = compute_features(atd)["tag_is_goal_stoppage"]  # loads the Smart Tagging XML
        force[goal] = mh.FORCE_DEAD
        n_goal = int(goal.sum())
    except Exception as e:  # Smart Tagging missing: kickoff constraints still apply
        print(f"  (no Smart Tagging goal stoppages: {e})")
    for k in (k1, k2):
        force[k:k + 15] = mh.FORCE_LIVE
    print(f"constraints: {int((force == mh.FORCE_DEAD).sum()):,} frames forced dead "
          f"({n_goal:,} from goal stoppages), {int((force == mh.FORCE_LIVE).sum())} forced live")
    return force


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "moe_model.json"))
    ap.add_argument("--name", default="submission_moe")
    ap.add_argument("--tune-decoder", type=int, default=0, metavar="TRIALS")
    ap.add_argument("--tune-keys", default="all",
                    help="comma-separated decoder keys to tune, e.g. m_live,switch (default: all 8)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-constraints", action="store_true",
                    help="decode without the kickoff / goal-stoppage constraints")
    args = ap.parse_args()

    t0 = time.time()
    atd = load_cached_atd()
    n = len(atd["ball_xy"])
    X, names, _ = mf.compute(atd)
    moe, prior, trained_names, dparams, meta = mh.load(args.model)
    if names != trained_names:
        sys.exit("feature list differs from the one the model was trained with -- retrain")
    proba = moe.predict_proba(X, mh.gate_inputs(X, names))
    gate = mh.restart_gate(X, names)
    print(f"features + emissions for {n:,} frames in {time.time() - t0:.0f}s "
          f"(model: {meta.get('trained_on')}, {meta.get('experts')} experts)")

    public = gt_codes("BAR-ATH_possession_gt_public.csv")
    hand = sparse_codes("user_manual_labels_sparse_5397_frames.csv", n)
    pub_runs = pm.valid_runs(public)
    force = None if args.no_constraints else match_constraints(atd, n)

    def evaluate(params, f=force):
        codes = mh.decode(proba, prior, GRADED_RUNS, n, params, force=f, gate=gate)
        return codes, score(public, codes, pub_runs)

    if force is not None:
        _, s0 = evaluate(dparams, None)
        print(f"public GT, no constraints:  {fmt(s0)}")
    codes, s = evaluate(dparams)
    print(f"public GT{', with constraints' if force is not None else ''}: {fmt(s)}")

    if args.tune_decoder:
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        from train_moe import sample_params
        rng = np.random.default_rng(args.seed)
        keys = None if args.tune_keys == "all" else set(args.tune_keys.split(","))
        base = dict(dparams)

        def restrict(c):
            return c if keys is None else {k: (c[k] if k in keys else base[k]) for k in base}

        best_p, best = dparams, s["spellF1"]
        # decode only the public window + margin while searching (fast)
        a, b = 11185 - 0, 18685
        sub_runs = [(0, b - a)]
        sub_gt = public[a:b]
        for i in range(args.tune_decoder):
            cand = restrict(sample_params(rng) if i < args.tune_decoder // 2 else
                            sample_params(rng, best_p, scale=1.0 - i / args.tune_decoder))
            c = mh.decode(proba[a:b], prior, sub_runs, b - a, cand,
                          force=None if force is None else force[a:b], gate=gate[a:b])
            v = score(sub_gt, c, sub_runs)["spellF1"]
            if v > best:
                best, best_p = v, cand
        dparams = best_p
        codes, s = evaluate(dparams)
        print(f"public GT, decoder tuned ON THE PUBLIC WINDOW (diagnostic only, not a valid "
              f"estimate): {fmt(s)}\n  params: {dparams}")

    hs = score(hand, codes, pm.valid_runs(hand))
    print(f"hand labels (overlap the public window): spellF1={hs['spellF1']:.4f} "
          f"macroF1={hs['macroF1']:.4f}")

    full = mh.decode(proba, prior, GRADED_RUNS, n, dparams, force=force, gate=gate)
    sp = pm.possession_spells(pm.fill_abstentions(full, GRADED_RUNS), GRADED_RUNS)
    print(f"full match: {len(sp)} possessions, mean {np.mean([(b - a) / 25 for a, b, _ in sp]):.1f}s, "
          f"dead-ball share {np.mean(np.concatenate([full[a:b] for a, b in GRADED_RUNS]) == 2):.1%}")

    name = args.name + ("_DIAGNOSTIC_public_tuned" if args.tune_decoder else "")
    out = os.path.join(ROOT, "evaluation", "results", f"{name}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{lbl}\n" for i, lbl in enumerate(LABEL[full]))
    reason = pm.validate_submission(pm.read_label_csv(out), n)
    print(f"wrote {out} -- validation {'FAILED: ' + reason if reason else 'passed'}")


if __name__ == "__main__":
    main()
