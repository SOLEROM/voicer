#!/usr/bin/env python3
"""
verify_voice_rknn.py – Compare two WAV files using RKNN-based ReDimNet model.

Usage:
    python verify_voice_rknn.py model.rknn audio1.wav audio2.wav [rk3588]
"""

import os
import sys
import numpy as np
import soundfile as sf
from scipy.signal import get_window, resample
from numpy.fft import rfft
from rknn.api import RKNN

# ────────────────────────────────────────────────────────────────
# CONFIG + DSP HELPERS
# ────────────────────────────────────────────────────────────────

EPS = 1e-8

def preemphasis(wave, alpha=0.97):
    out = wave.copy()
    out[1:] -= alpha * wave[:-1]
    return out

def hz_to_mel(hz): return 2595 * np.log10(1 + hz / 700.0)
def mel_to_hz(mel): return 700 * (10**(mel / 2595.0) - 1)

def mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts  = mel_to_hz(mel_pts)
    bins    = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        right = min(right, fb.shape[1] - 1)
        fb[i - 1, left:center] = (np.arange(left, center) - left) / max(center - left, 1)
        fb[i - 1, center:right] = (right - np.arange(center, right)) / max(right - center, 1)
    return fb

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
    if do_preemph:
        wave = preemphasis(wave, alpha=0.97)

    pad = n_fft // 2
    wave_padded = np.pad(wave, (pad, pad), mode="reflect")

    win = get_window("hamming", win_len, fftbins=True).astype(np.float32)
    win = np.pad(win, (0, n_fft - win_len))

    frames = []
    for start in range(0, len(wave_padded) - n_fft + 1, hop):
        frame = wave_padded[start:start + n_fft] * win
        spec = np.abs(rfft(frame, n=n_fft)) ** 2
        frames.append(spec)

    if not frames:
        raise RuntimeError("❌ Audio too short for one FFT frame")

    spec = np.stack(frames, axis=1)
    mel_fb = mel_filterbank(sr, n_fft, n_mels, f_min, f_max)
    mel = mel_fb @ spec

    logmel = np.log(mel + EPS)

    T = logmel.shape[1]
    if T < target_T:
        logmel = np.pad(logmel, ((0, 0), (0, target_T - T)), mode="constant")
    elif T > target_T:
        start = (T - target_T) // 2
        logmel = logmel[:, start:start + target_T]

    logmel = logmel - logmel.mean(axis=1, keepdims=True)

    return logmel[np.newaxis, np.newaxis, :, :].astype(np.float32)

# ────────────────────────────────────────────────────────────────
# EMBEDDING + SIMILARITY
# ────────────────────────────────────────────────────────────────

def extract_embedding(rknn: RKNN, wav_path: str) -> np.ndarray:
    waveform, sr = sf.read(wav_path)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    TARGET_SR = 16000
    if sr != TARGET_SR:
        print(f"[INFO] Resampling from {sr} Hz → {TARGET_SR} Hz...")
        duration = len(waveform) / sr
        waveform = resample(waveform, int(duration * TARGET_SR))
        sr = TARGET_SR

    logmel_nchw = compute_logmel(waveform, sr=sr)  # [1, 1, 60, 200]
    logmel_nhwc = np.transpose(logmel_nchw, (0, 2, 3, 1))  # [1, 60, 200, 1]

    output = rknn.inference(inputs=[logmel_nhwc.astype(np.float32)])
    return output[0]

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS)

# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────

def main(model_path: str, wav1: str, wav2: str, target: str = "rk3588"):
    print(f"[1/4] Loading RKNN model: {model_path}")
    rknn = RKNN()
    if rknn.load_rknn(model_path) != 0:
        raise RuntimeError("❌ Failed to load RKNN model")

    print(f"[2/4] Initializing runtime for target: {target}")
    if rknn.init_runtime(target=target) != 0:
        raise RuntimeError("❌ Failed to initialize RKNN runtime")

    print(f"[3/4] Extracting embeddings for:\n  {wav1}\n  {wav2}")
    emb1 = extract_embedding(rknn, wav1)
    emb2 = extract_embedding(rknn, wav2)

    rknn.release()

    print("[4/4] Computing cosine similarity…")
    sim = cosine_similarity(emb1, emb2)
    print(f"\n✅ Cosine Similarity: {sim:.4f}")
    print(f"🔎 Distance (1 - sim): {1 - sim:.4f}")

# ────────────────────────────────────────────────────────────────
# CLI ENTRY
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    model_path = sys.argv[1]
    wav1 = sys.argv[2]
    wav2 = sys.argv[3]
    target = sys.argv[4] if len(sys.argv) > 4 else "rk3588"
    main(model_path, wav1, wav2, target)
