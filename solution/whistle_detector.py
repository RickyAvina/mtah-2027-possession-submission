"""Referee whistle detector from tactical video audio track.

Extracts the audio track from BAR-ATH_tactical.mp4 using PyAV and performs
spectro-temporal peak detection in the referee whistle band (2.7 kHz - 3.8 kHz).
Generates frame-level whistle confidence / detection indicators.
"""

import os
import sys
import time
import numpy as np
from scipy.signal import spectrogram
import av

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

CACHE_DIR = os.path.join(ROOT, "cache")
WHISTLE_CACHE_FILE = os.path.join(CACHE_DIR, "whistle_detections.npz")


def extract_whistle_frames(video_path=None, force=False):
    """Scan audio for whistle events and return frame-level indicator (N_frames,) boolean array."""
    if os.path.exists(WHISTLE_CACHE_FILE) and not force:
        print(f"Loading cached whistle detections from {WHISTLE_CACHE_FILE} ...")
        cached = np.load(WHISTLE_CACHE_FILE)
        return cached["is_whistle"], cached["whistle_ratio"]

    if video_path is None:
        from games import GAME, require

        full_video_path = str(require(["video"], GAME, None, "tactical")[0])
    else:
        full_video_path = os.path.join(ROOT, video_path)
    if not os.path.exists(full_video_path):
        raise FileNotFoundError(f"Video file not found: {full_video_path}")

    print(f"Extracting audio and detecting referee whistles from {full_video_path} ...")
    t0 = time.time()
    container = av.open(full_video_path)
    audio_stream = [s for s in container.streams if s.type == "audio"][0]
    sr = audio_stream.rate

    # Decode audio in chunks
    pcm_chunks = []
    chunk_times = []
    for frame in container.decode(audio_stream):
        t = float(frame.pts * frame.time_base) if frame.pts else 0.0
        mono = frame.to_ndarray().mean(axis=0)  # mix down to mono
        pcm_chunks.append(mono)
        chunk_times.append(t)

    pcm = np.concatenate(pcm_chunks)
    total_audio_s = len(pcm) / float(sr)
    n_frames = int(round(total_audio_s * 25.0))
    print(f"Decoded {total_audio_s:.1f}s of audio ({len(pcm)} samples at {sr} Hz) in {time.time() - t0:.2f}s")

    # Spectrogram computation
    # nperseg=1024 -> ~21.3ms window, noverlap=512 -> ~10.6ms hop
    f, t_spec, Sxx = spectrogram(pcm, fs=sr, nperseg=1024, noverlap=512)

    whistle_mask = (f >= 2600) & (f <= 3900)
    whistle_power = Sxx[whistle_mask, :].sum(axis=0)
    total_power = Sxx.sum(axis=0) + 1e-9
    ratio = whistle_power / total_power

    # Map time slices to 25 fps video frames (0..199318)
    n_video_frames = 199319
    whistle_ratio = np.zeros(n_video_frames, dtype=np.float32)
    is_whistle = np.zeros(n_video_frames, dtype=bool)

    # Frame mapping
    frame_indices = np.clip(np.round(t_spec * 25.0).astype(int), 0, n_video_frames - 1)
    for idx, r in zip(frame_indices, ratio):
        if r > whistle_ratio[idx]:
            whistle_ratio[idx] = r

    # Threshold: ratio > 0.32 indicates distinct whistle burst
    is_whistle = whistle_ratio > 0.32

    # Save cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(WHISTLE_CACHE_FILE, is_whistle=is_whistle, whistle_ratio=whistle_ratio)
    print(f"Saved whistle cache with {is_whistle.sum()} detected whistle frames to {WHISTLE_CACHE_FILE}")

    return is_whistle, whistle_ratio


if __name__ == "__main__":
    is_whistle, ratio = extract_whistle_frames()
    print("Whistle frames in first 20,000 frames:")
    whistle_idx = np.where(is_whistle[:20000])[0]
    print(whistle_idx)
