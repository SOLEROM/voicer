#!/usr/bin/env python3
"""
rknn_embed_or_verify.py — ReDimNet-NoMel RKNN front-end + inference

Usage
-----
# 1) Single-file inference (prints embedding stats; optionally save)
python rknn_embed_or_verify.py  model.rknn  audio.wav  [--target rk3588] [--out emb.npy]

# 2) Two-file verification (prints cosine similarity)
python rknn_embed_or_verify.py  model.rknn  audio1.wav  audio2.wav  [--target rk3588]
"""

import os
import sys
import argparse
import numpy as np
import soundfile as sf
from scipy.signal import get_window, resample
from numpy.fft import rfft
from rknn.api import RKNN

# RKNN log level (3 = warnings and up)
os.environ["RKNN_LOG_LEVEL"] = "3"

# ───────────────────────── Front-end constants (IDRnD) ─────────────────────────
_PREEMPH  = 0.97
_SR       = 16_000
_N_FFT    = 512
_WIN_LEN  = 400
_HOP      = 240
_N_MELS   = 60
_F_MIN    = 20.0
_F_MAX    = 7_600.0
_TARGET_T = 134       # target number of frames expected by the model
_EPS      = 1e-6

# ─────────────────────────── DSP helper functions ────────────────────────────
def preemphasis(wave: np.ndarray, alpha: float = _PREEMPH) -> np.ndarray:
    out = wave.astype(np.float32, copy=True)
    if out.size > 1:
        out[1:] -= alpha * out[:-1]
    return out

def hz_to_mel(hz):
    hz = np.asanyarray(hz, dtype=np.float64)
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel):
    mel = np.asanyarray(mel, dtype=np.float64)
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def mel_filterbank(sr=_SR, n_fft=_N_FFT, n_mels=_N_MELS,
                   fmin=_F_MIN, fmax=_F_MAX) -> np.ndarray:
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts  = mel_to_hz(mel_pts)
    bins    = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        right = min(right, fb.shape[1] - 1)
        if center > left:
            fb[i - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            fb[i - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    return fb.astype(np.float16)

# Precompute filterbank and window once
_MEL_FB = mel_filterbank()
_WINDOW = np.pad(get_window("hamming", _WIN_LEN, fftbins=True).astype(np.float32),
                 (0, _N_FFT - _WIN_LEN))

# ───────────────────────────── Wave → log-Mel ───────────────────────────────
def waveform_to_logmel(wave: np.ndarray) -> np.ndarray:
    """
    Input:  wave (N,) mono 16k PCM
    Output: (1, 1, 60, _TARGET_T) float16 (NCHW)
    """
    # 1) pre-emphasis
    wave = preemphasis(wave).astype(np.float32)

    # 2) reflection pad (centered STFT like torch.stft(center=True))
    pad = _N_FFT // 2
    wave_padded = np.pad(wave, (pad, pad), mode="reflect")

    # 3) STFT power
    frames = []
    for start in range(0, len(wave_padded) - _N_FFT + 1, _HOP):
        frame = wave_padded[start:start + _N_FFT] * _WINDOW
        spec = np.abs(rfft(frame, n=_N_FFT)) ** 2
        frames.append(spec.astype(np.float32))
    if not frames:
        raise RuntimeError("❌ Audio too short for one FFT frame")

    spec = np.stack(frames, axis=1)  # [freq, T]

    # 4) Mel projection
    mel = _MEL_FB @ spec             # [60, T]

    # 5) log + mean norm
    logmel = np.log(mel + _EPS)
    logmel -= logmel.mean(axis=1, keepdims=True)

    # 6) pad/crop to target T
    T = logmel.shape[1]
    if T < _TARGET_T:
        logmel = np.pad(logmel, ((0, 0), (0, _TARGET_T - T)), mode="constant")
        print(f"[INFO] Padding log-mel {T} → {_TARGET_T} frames")
    elif T > _TARGET_T:
        start = (T - _TARGET_T) // 2
        logmel = logmel[:, start:start + _TARGET_T]
        print(f"[INFO] Cropping log-mel {T} → {_TARGET_T} frames")

    # [1,1,60,T] float16
    return logmel[np.newaxis, np.newaxis, :, :].astype(np.float16)

# ───────────────────────────── Audio I/O helper ─────────────────────────────
def load_mono_16k(path: str) -> np.ndarray:
    wave, sr = sf.read(path, always_2d=False)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if sr != _SR:
        # resample by length ratio to preserve duration
        target_len = int(len(wave) * (_SR / sr))
        print(f"[INFO] Resampling {sr} → {_SR} Hz (len {len(wave)} → {target_len})")
        wave = resample(wave, target_len)
    return wave.astype(np.float32, copy=False)

# ───────────────────────────── RKNN helpers ────────────────────────────────
def init_rknn(model_path: str, target: str) -> RKNN:
    print(f"[1/3] Loading RKNN model: {model_path}")
    rk = RKNN()
    if rk.load_rknn(model_path) != 0:
        raise RuntimeError("❌ Failed to load RKNN file")
    print(f"[2/3] Initialising runtime: {target}")
    if rk.init_runtime(target=target) != 0:
        raise RuntimeError("❌ Failed to initialise RKNN runtime")
    return rk

def infer_embedding(rk: RKNN, wav_path: str) -> np.ndarray:
    wave = load_mono_16k(wav_path)
    logmel = waveform_to_logmel(wave)         # [1,1,60,_TARGET_T], float16
    out = rk.inference(inputs=[logmel], data_format='nchw')
    emb = out[0]
    return np.asarray(emb)

# ───────────────────────────── Similarity ──────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + _EPS
    return float(np.dot(a, b) / denom)

# ──────────────────────────────── Main ─────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="RKNN inference for ReDimNet-NoMel (single embed or pairwise verify)")
    ap.add_argument("model", help="Path to .rknn model file")
    ap.add_argument("wav1",  help="Path to first WAV file")
    ap.add_argument("wav2",  nargs="?", help="Optional second WAV (enables verification)")
    ap.add_argument("--target", default="rk3588", help="RKNN target (default: rk3588)")
    ap.add_argument("--out", help="When using single-file mode, save embedding to .npy")
    args = ap.parse_args()

    rk = init_rknn(args.model, args.target)

    try:
        if args.wav2 is None:
            # Single-file inference
            print("[3/3] Running embedding inference …")
            emb = infer_embedding(rk, args.wav1)
            rk.release()

            print(f"\n✅ Embedding produced: shape={emb.shape}, dtype={emb.dtype}")
            print(f"   min={emb.min():.6f}  max={emb.max():.6f}")
            flat = emb.ravel()
            print("   first 10 values:", np.array2string(flat[:10], precision=6, separator=", "))

            if args.out:
                np.save(args.out, emb)
                print(f"💾 Saved embedding to {args.out}")
        else:
            # Two-file verification
            print("[3/3] Extracting embeddings …")
            emb1 = infer_embedding(rk, args.wav1)
            emb2 = infer_embedding(rk, args.wav2)
            rk.release()

            sim = cosine_similarity(emb1, emb2)
            print("\n✅ Cosine Similarity:", f"{sim:.6f}")
            print("🔎 Distance (1 - sim):", f"{1.0 - sim:.6f}")

    finally:
        # Safety in case of early exceptions
        try:
            rk.release()
        except Exception:
            pass

if __name__ == "__main__":
    main()
