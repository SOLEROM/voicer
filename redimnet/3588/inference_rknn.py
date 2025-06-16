#!/usr/bin/env python3
"""
inference_rknn.py – Rockchip RKNN inference for ReDimNet checkpoints

Usage:
    python inference_rknn.py  model.rknn  audio.wav
"""

import os
import sys
import math
import numpy as np
import soundfile as sf
from scipy.signal import get_window, resample
from numpy.fft    import rfft
from rknn.api     import RKNN

# ----------------------------------------------------------------------
#  Global RKNN log level
# ----------------------------------------------------------------------
os.environ["RKNN_LOG_LEVEL"] = "3"

# ----------------------------------------------------------------------
#  DSP HELPERS
# ----------------------------------------------------------------------
EPS = 1e-8           # small constant to avoid /0 and log(0)


def preemphasis(wave, alpha: float = 0.97) -> np.ndarray:
    """ y[n] = x[n] − α·x[n−1]   (first sample unchanged) """
    out = wave.copy()
    out[1:] -= alpha * wave[:-1]
    return out


def hz_to_mel(hz):            # HTK formula
    return 2595.0 * np.log10(1 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10 ** (mel / 2595.0) - 1)


def mel_filterbank(sr, n_fft, n_mels, f_min, f_max):
    """Returns an (n_mels × (n_fft//2+1)) triangular filterbank."""
    mel_pts = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_pts  = mel_to_hz(mel_pts)
    bins    = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        right = min(right, fb.shape[1] - 1)        # safety clip

        # rising slope
        fb[i - 1, left:center] = (
            np.arange(left, center) - left) / max(center - left, 1)
        # falling slope
        fb[i - 1, center:right] = (
            right - np.arange(center, right)) / max(right - center, 1)
    return fb


# ----------------------------------------------------------------------
#  LOG-MEL GENERATOR  (matches ReDimNet “MelBanks” parameters)
# ----------------------------------------------------------------------
def compute_logmel(
    wave: np.ndarray,
    sr: int = 16000,
    n_fft: int = 512,
    win_len: int = 400,
    hop: int = 160,
    n_mels: int = 60,
    f_min: float = 20.,
    f_max: float = 7600.,
    target_T: int = 200,
    do_preemph: bool = True,
) -> np.ndarray:
    """
    Return shape: [1, 1, n_mels, target_T] — float32
    """

    # 1) Optional pre-emphasis
    if do_preemph:
        wave = preemphasis(wave, alpha=0.97)

    # 2) Padding for centered STFT
    pad = n_fft // 2
    wave_padded = np.pad(wave, (pad, pad), mode="reflect")

    # 3) Create window (Hamming like PyTorch)
    win = get_window("hamming", win_len, fftbins=True).astype(np.float32)
    win = np.pad(win, (0, n_fft - win_len))  # zero-pad to match n_fft

    # 4) Frame-by-frame STFT power
    frames = []
    for start in range(0, len(wave_padded) - n_fft + 1, hop):
        frame = wave_padded[start:start + n_fft] * win
        spec  = np.abs(rfft(frame, n=n_fft)) ** 2
        frames.append(spec)

    if not frames:
        raise RuntimeError("❌ Audio too short for one FFT frame")

    spec = np.stack(frames, axis=1)  # shape: [n_freq_bins, frames]

    # 5) Mel filterbank
    mel_fb = mel_filterbank(sr, n_fft, n_mels, f_min, f_max)
    mel    = mel_fb @ spec  # shape: [n_mels, frames]

    # 6) Log scale (natural log, not dB)
    logmel = np.log(mel + EPS)  # match PyTorch's torchaudio behavior

    # 7) Crop or pad to target frame count
    T = logmel.shape[1]
    if T < target_T:
        logmel = np.pad(logmel, ((0, 0), (0, target_T - T)), mode="constant")
    elif T > target_T:
        start = (T - target_T) // 2
        logmel = logmel[:, start:start + target_T]

    # 8) Mean normalize per channel (no CMVN, just subtract mean like torch)
    logmel = logmel - logmel.mean(axis=1, keepdims=True)

    # 9) Return as [1, 1, n_mels, target_T] float32
    return logmel[np.newaxis, np.newaxis, :, :].astype(np.float32)


# ----------------------------------------------------------------------
#  RKNN INFERENCE
# ----------------------------------------------------------------------
def run_inference_rknn(rknn_path: str, wav_path: str):
    print("[1/4] Loading RKNN model …")
    rknn = RKNN()
    if rknn.load_rknn(rknn_path) != 0:
        raise RuntimeError("❌ Failed to load RKNN file")

    print("[2/4] Initialising runtime …")
    if rknn.init_runtime(target="rk3588") != 0:
        raise RuntimeError("❌ Failed to initialise RKNN runtime")

    # ------------------------------------------------------------------
    #  AUDIO PRE-PROCESS
    # ------------------------------------------------------------------
    print("[3/4] Pre-processing audio …")
    wave, sr = sf.read(wav_path)                     # ndarray, shape (N,) or (N, C)
    if wave.ndim > 1:                               # stereo → mono
        wave = wave.mean(axis=1)
    wave = wave.astype(np.float32)

    # Resample if needed
    TARGET_SR = 16000
    if sr != TARGET_SR:
        print(f"[INFO] Resampling {sr} → {TARGET_SR} Hz")
        duration = len(wave) / sr
        wave = resample(wave, int(duration * TARGET_SR))
        sr = TARGET_SR

    logmel_nchw = compute_logmel(wave, sr=sr)       # [1,1,60,200]
    logmel_nhwc = np.transpose(logmel_nchw, (0, 2, 3, 1))  # [1,60,200,1]

    # ------------------------------------------------------------------
    #  INFERENCE
    # ------------------------------------------------------------------
    print("[4/4] Running inference …")
    outputs = rknn.inference(inputs=[logmel_nhwc])
    rknn.release()

    emb = outputs[0]
    print("\n[DEBUG] Output embedding stats  "
          f"(shape {emb.shape}):  min={emb.min():.3f}  max={emb.max():.3f}")
    print("Preview:", emb[0][:10])

    return emb


# ----------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    run_inference_rknn(sys.argv[1], sys.argv[2])
