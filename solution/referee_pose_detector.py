"""Generic Main Referee Tracker & Self-Calibrated 2D Pose Dead-Ball Signal Detector.

Architecture:
1. Untracked Pitch-Interior Human Detection (Homography + Tracked Player Masking):
   - Projects all 22 tracked players (`corrected_xy`) onto the tactical video frame via homography `H[t]`.
   - Any person detected on the pitch interior whose foot projection (`unproject(H[t], u_foot, v_foot)`)
     is separated from all tracked players is a Main Referee candidate.
   - Automatically learns the Main Referee's kit HSV color profile from isolated detections
     and tracks the referee smoothly across frames.
2. Self-Calibrated Scale-Invariant 2D Referee Pose Extraction (`yolov8n-pose.pt`):
   - Crops and upscales the region around the Main Referee and extracts 17 COCO keypoints.
   - Computes three raw scale-invariant pose indicators normalized by torso height `h_torso = y_hip - y_shoulder`:
     * `ref_arm_elev`: `(y_shoulder - min_y(wrists, elbows)) / h_torso`
     * `ref_wrist_to_head`: `min_dist(wrist, nose) / h_torso`
     * `ref_arm_span`: `||L_wrist - R_wrist|| / h_torso`
   - Standardizes each indicator into an empirical z-score `(x - median) / (1.4826 * MAD)` calibrated
     on non-holdout match frames (strictly outside `[11185, 18685)`), and maps the maximum gesture
     z-score above the standard 95% normal confidence bound (`z_0 = 1.96`) through a logistic sigmoid.
"""

import os
import sys
import time
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from starter.load_homography import load as load_homography, project, unproject
from games import GAME, require

# Standard two-sided 95% normal confidence quantile Phi^{-1}(0.975)
Z_CONF_95 = 1.959964


def compute_pose_metrics_from_keypoints(kp):
    """Compute raw scale-invariant referee pose indicators from (17, 3) COCO keypoints."""
    y_sh = 0.5 * float(kp[5, 1] + kp[6, 1])
    y_hip = 0.5 * float(kp[11, 1] + kp[12, 1])
    torso_h = max(3.0, y_hip - y_sh)

    lw_elev = (y_sh - float(kp[9, 1])) / torso_h
    rw_elev = (y_sh - float(kp[10, 1])) / torso_h
    le_elev = (y_sh - float(kp[7, 1])) / torso_h
    re_elev = (y_sh - float(kp[8, 1])) / torso_h
    max_arm_elev = float(max(lw_elev, rw_elev, le_elev, re_elev))

    arm_span = float(np.linalg.norm(kp[9, :2] - kp[10, :2]) / torso_h)

    d_lw_nose = float(np.linalg.norm(kp[9, :2] - kp[0, :2]) / torso_h)
    d_rw_nose = float(np.linalg.norm(kp[10, :2] - kp[0, :2]) / torso_h)
    wrist_to_head = float(min(d_lw_nose, d_rw_nose))

    return max_arm_elev, wrist_to_head, arm_span


def calibrate_and_score_referee_pose(
    ref_xy,
    ref_arm_elev,
    ref_wrist_to_head,
    ref_arm_span,
    holdout_intervals=None,
    fps=25,
):
    """Self-calibrate referee posture z-scores on non-holdout frames and return smoothed probability."""
    n_frames = len(ref_arm_elev)
    if holdout_intervals is None:
        holdout_intervals = [(11185, 18685)]

    valid = ~np.isnan(ref_xy[:, 0])
    cal_mask = valid.copy()
    for h_s, h_e in holdout_intervals:
        cal_mask[max(0, h_s) : min(n_frames, h_e)] = False
    if not np.any(cal_mask):
        cal_mask = valid

    def robust_stats(arr, mask):
        vals = arr[mask]
        if len(vals) == 0:
            return 0.0, 1.0
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)) * 1.4826)
        return med, max(mad, 1e-3)

    med_elev, std_elev = robust_stats(ref_arm_elev, cal_mask)
    med_w2h, std_w2h = robust_stats(ref_wrist_to_head, cal_mask)
    med_span, std_span = robust_stats(ref_arm_span, cal_mask)

    z_elev = (ref_arm_elev - med_elev) / std_elev
    arm_active = (z_elev > 0.0).astype(np.float32)
    z_w2h = ((med_w2h - ref_wrist_to_head) / std_w2h) * arm_active
    z_span = ((ref_arm_span - med_span) / std_span) * arm_active

    z_max = np.maximum.reduce([z_elev, z_w2h, z_span])
    ref_raw_signal = np.where(
        valid,
        1.0 / (1.0 + np.exp(-np.clip(z_max - Z_CONF_95, -15.0, 15.0))),
        0.0,
    ).astype(np.float32)

    # Smooth over 0.4s (11 frames at 25fps) to suppress single-frame keypoint noise
    w_len = max(3, int(round(0.4 * fps)) | 1)
    k_smooth = np.ones(w_len, dtype=np.float32) / float(w_len)
    ref_signal_score = np.convolve(ref_raw_signal, k_smooth, mode="same").astype(np.float32)

    return ref_raw_signal, ref_signal_score


def extract_referee_pose_features(
    corrected_xy,
    ball_xy_interp,
    pitch_length=105.0,
    pitch_width=68.0,
    intervals=None,
    holdout_intervals=None,
    step=2,
    cache_path=None,
    force_recompute=False,
):
    """Extract or load cached Main Referee trajectory and self-calibrated 2D pose features."""
    if cache_path is None:
        cache_path = os.path.join(ROOT, "cache/referee_pose_features.npz")

    n_frames = len(ball_xy_interp)
    if os.path.exists(cache_path) and not force_recompute:
        data = np.load(cache_path)
        if len(data["ref_signal_score"]) == n_frames:
            return {k: data[k] for k in data.files}

    if intervals is None:
        # Include non-public calibration segment [18685, 21185) alongside target intervals
        intervals = [(11185, 21185)]

    import torch
    from ultralytics import YOLO

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    weights_path = os.path.join(ROOT, "yolov8n-pose.pt")
    pose_model = YOLO(weights_path)

    video_path = require(["video"], GAME, None, "tactical")[0]
    H_tac, _, valid_tac = load_homography(source="tactical")

    ref_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)
    ref_arm_elev = np.full(n_frames, -0.5, dtype=np.float32)
    ref_wrist_to_head = np.full(n_frames, 1.3, dtype=np.float32)
    ref_arm_span = np.full(n_frames, 0.7, dtype=np.float32)

    cap = cv2.VideoCapture(video_path)
    t0 = time.time()

    for s_f, e_f in intervals:
        print(f"Extracting Main Referee 2D pose on [{s_f}, {e_f}) (step={step}, device={device})...")
        last_ref_xy = None
        missed_steps = 0
        ref_hsv_proto = None

        cap.set(cv2.CAP_PROP_POS_FRAMES, s_f)
        curr_f = s_f

        for t in range(s_f, e_f, step):
            while curr_f <= t:
                ret, frame = cap.read()
                curr_f += 1
            if not ret or not valid_tac[t]:
                continue

            H = H_tac[t]
            p_xy = corrected_xy[t]
            vis = ~np.isnan(p_xy[:, 0])
            if not np.any(vis):
                continue

            bu, bv = project(H, ball_xy_interp[t, 0] / pitch_length, ball_xy_interp[t, 1] / pitch_width)
            pu, pv = project(H, p_xy[vis, 0] / pitch_length, p_xy[vis, 1] / pitch_width)

            if last_ref_xy is not None and missed_steps <= 4:
                ru, rv = project(H, last_ref_xy[0] / pitch_length, last_ref_xy[1] / pitch_width)
                cu, cv_pt = int(ru), int(rv)
                half_w, half_h = 180, 140
                imgsz = 640
            else:
                cu = int(0.6 * bu + 0.4 * np.median(pu))
                cv_pt = int(0.6 * bv + 0.4 * np.median(pv))
                half_w, half_h = 360, 240
                imgsz = 960

            u1, u2 = max(0, cu - half_w), min(frame.shape[1], cu + half_w)
            v1, v2 = max(0, cv_pt - half_h), min(frame.shape[0], cv_pt + half_h)
            if u2 - u1 < 40 or v2 - v1 < 40:
                continue

            scale_up = 2.5
            crop = frame[v1:v2, u1:u2]
            crop_up = cv2.resize(crop, (0, 0), fx=scale_up, fy=scale_up, interpolation=cv2.INTER_LINEAR)

            res = pose_model.predict(crop_up, imgsz=imgsz, conf=0.10, device=device, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
                missed_steps += 1
                continue

            boxes = res.boxes.xyxy.cpu().numpy() / scale_up
            kpts = res.keypoints.data.cpu().numpy()
            kpts[:, :, :2] /= scale_up

            best_cand = None
            for box, kp in zip(boxes, kpts):
                x1, y1, x2, y2 = box
                u_foot = u1 + 0.5 * (x1 + x2)
                v_foot = v1 + y2
                nx, ny = unproject(H, u_foot, v_foot)
                px_m, py_m = float(nx * pitch_length), float(ny * pitch_width)

                if not (2.0 <= px_m <= pitch_length - 2.0 and 2.0 <= py_m <= pitch_width - 2.0):
                    continue

                d_p = float(np.min(np.linalg.norm(p_xy[vis] - np.array([px_m, py_m]), axis=1)))
                d_ball = float(np.linalg.norm(ball_xy_interp[t] - np.array([px_m, py_m])))

                tx1, tx2 = int(u1 + x1 + 0.25 * (x2 - x1)), int(u1 + x2 - 0.25 * (x2 - x1))
                ty1, ty2 = int(v1 + y1 + 0.20 * (y2 - y1)), int(v1 + y1 + 0.55 * (y2 - y1))
                torso = frame[max(0, ty1) : min(frame.shape[0], ty2), max(0, tx1) : min(frame.shape[1], tx2)]
                if torso.size > 0:
                    hsv_med = np.median(cv2.cvtColor(torso, cv2.COLOR_BGR2HSV).reshape(-1, 3), axis=0)
                else:
                    hsv_med = np.zeros(3)

                color_cost = 0.0
                if ref_hsv_proto is not None and torso.size > 0:
                    dh = min(abs(hsv_med[0] - ref_hsv_proto[0]), 180.0 - abs(hsv_med[0] - ref_hsv_proto[0]))
                    ds = abs(hsv_med[1] - ref_hsv_proto[1])
                    color_cost = 0.08 * dh + 0.02 * ds

                if last_ref_xy is not None and missed_steps <= 6:
                    d_prev = float(np.linalg.norm(np.array([px_m, py_m]) - last_ref_xy))
                    max_jump = 1.5 + 0.8 * (missed_steps + 1)
                    if d_prev <= max_jump and (d_p > 1.2 or color_cost < 2.5):
                        score = d_prev + color_cost - 0.3 * min(d_p, 4.0)
                        if best_cand is None or score < best_cand[0]:
                            best_cand = (score, px_m, py_m, kp, hsv_med, d_p)
                else:
                    if d_p > 1.8 and d_ball < 28.0:
                        score = d_ball + color_cost
                        if best_cand is None or score < best_cand[0]:
                            best_cand = (score, px_m, py_m, kp, hsv_med, d_p)

            if best_cand is not None:
                _, rx, ry, kp, hsv_med, d_p = best_cand
                last_ref_xy = np.array([rx, ry], dtype=np.float32)
                missed_steps = 0
                if d_p > 2.5 and hsv_med[1] > 40:
                    if ref_hsv_proto is None:
                        ref_hsv_proto = hsv_med.copy()
                    else:
                        ref_hsv_proto = 0.9 * ref_hsv_proto + 0.1 * hsv_med

                elev, w2h, span = compute_pose_metrics_from_keypoints(kp)
                for fill_t in range(t, min(e_f, t + step)):
                    ref_xy[fill_t] = last_ref_xy
                    ref_arm_elev[fill_t] = elev
                    ref_wrist_to_head[fill_t] = w2h
                    ref_arm_span[fill_t] = span
            else:
                missed_steps += 1
                if last_ref_xy is not None and missed_steps <= 3:
                    for fill_t in range(t, min(e_f, t + step)):
                        ref_xy[fill_t] = last_ref_xy

    cap.release()

    ref_raw_signal, ref_signal_score = calibrate_and_score_referee_pose(
        ref_xy, ref_arm_elev, ref_wrist_to_head, ref_arm_span, holdout_intervals=holdout_intervals
    )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        ref_xy=ref_xy,
        ref_arm_elev=ref_arm_elev,
        ref_wrist_to_head=ref_wrist_to_head,
        ref_arm_span=ref_arm_span,
        ref_raw_signal=ref_raw_signal,
        ref_signal_score=ref_signal_score,
    )
    print(f"Saved self-calibrated referee pose features to {cache_path} in {time.time() - t0:.1f}s")
    return {
        "ref_xy": ref_xy,
        "ref_arm_elev": ref_arm_elev,
        "ref_wrist_to_head": ref_wrist_to_head,
        "ref_arm_span": ref_arm_span,
        "ref_raw_signal": ref_raw_signal,
        "ref_signal_score": ref_signal_score,
    }
