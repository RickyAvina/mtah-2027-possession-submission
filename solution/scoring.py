"""Thin wrapper over the official scorer for int-coded labels (0=H, 1=A, 2=D, -1/3=X)."""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import evaluation.possession_metrics as pm  # noqa: E402


def to_pm(codes):
    c = np.asarray(codes).astype(np.int64).copy()
    c[(c < 0) | (c > 2)] = pm.X
    return c


def score(gt, pred, runs=None):
    gt, pred = to_pm(gt), to_pm(pred)
    if runs is None:
        runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    cm3 = pm.scored_confusion(pm.confusion(gt, pred))
    f1s = [pm.per_class_prf(cm3, k)[2] for k in (pm.H, pm.A, pm.D)]
    sp = pm.spell_scores(gt, filled, runs)
    return {
        "spellF1": sp["mean_f1"],
        "macroF1": float(np.mean(f1s)),
        "f1_H": f1s[0], "f1_A": f1s[1], "f1_D": f1s[2],
        "n_gt": sp["n_gt"], "n_pred": sp["n_pred"],
        "f1@0.5": sp["per_iou"][0.5]["f1"],
    }


def fmt(s):
    return (f"spellF1={s['spellF1']:.4f} macroF1={s['macroF1']:.4f} "
            f"(H {s['f1_H']:.3f} A {s['f1_A']:.3f} D {s['f1_D']:.3f}) "
            f"spells pred/gt={s['n_pred']}/{s['n_gt']}")
