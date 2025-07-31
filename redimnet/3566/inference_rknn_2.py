#!/usr/bin/env python3
"""
verify_voice_rknn.py – Compare two WAV files with a ReDimNetNoMel RKNN model
using an all-NumPy front-end (no PyTorch/torchaudio).

Usage
-----
python verify_voice_rknn.py  model.rknn  audio1.wav  audio2.wav  [rk3588]
"""

# ───────────────────────── Imports (PyTorch-free) ─────────────────────────
import os
import sys
import numpy as np
import soundfile as sf
from scipy.signal import get_window, resample
from numpy.fft    import rfft
from rknn.api     import RKNN

os.environ["RKNN_LOG_LEVEL"] = "3"       # warnings and up

# ─────────────────────── Front-end constants (IDRnD) ──────────────────────
_PREEMPH  = 0.97
_SR       = 16_000
_N_FFT    = 512
_WIN_LEN  = 400
_HOP      = 240
_N_MELS   = 60
_F_MIN    = 20.0
_F_MAX    = 7_600.0
_TARGET_T = 134
_EPS      = 1e-6

# ────────────────────────── DSP helper functions ──────────────────────────
def preemphasis(wave: np.ndarray, alpha: float = _PREEMPH) -> np.ndarray:
    out = wave.copy()
    out[1:] -= alpha * wave[:-1]
    return out


def hz_to_mel(hz):  return 2595.0 * np.log10(1.0 + hz / 700.0)
def mel_to_hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(sr=_SR, n_fft=_N_FFT, n_mels=_N_MELS,
                   fmin=_F_MIN, fmax=_F_MAX) -> np.ndarray:
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts  = mel_to_hz(mel_pts)
    bins    = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float16)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        right = min(right, fb.shape[1] - 1)

        if center > left:
            fb[i - 1, left:center] = (
                np.arange(left, center) - left) / (center - left)
        if right > center:
            fb[i - 1, center:right] = (
                right - np.arange(center, right)) / (right - center)
    return fb


# Pre-compute filterbank and window once
_MEL_FB = mel_filterbank()
_WINDOW = np.pad(
    get_window("hamming", _WIN_LEN, fftbins=True).astype(np.float16),
    (0, _N_FFT - _WIN_LEN)
)

# ────────────────────────── Wave → log-Mel front-end ───────────────────────
def compute_logmel(wave: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    wave : 1D np.ndarray, mono 16-kHz PCM

    Returns
    -------
    np.ndarray  shape (1, 1, 60, 200)  – NCHW float16
    """
    # 1) pre-emphasis
    wave = preemphasis(wave).astype(np.float16)

    # 2) reflection pad (centered STFT)
    pad = _N_FFT // 2
    wave = np.pad(wave, (pad, pad), mode="reflect")

    # 3) STFT power spectrum
    frames = []
    for start in range(0, len(wave) - _N_FFT + 1, _HOP):
        frame = wave[start:start + _N_FFT] * _WINDOW
        spec  = np.abs(rfft(frame, n=_N_FFT)) ** 2
        frames.append(spec)
    if not frames:
        raise RuntimeError("❌ Audio too short for even one FFT frame")

    spec = np.stack(frames, axis=1)                    # [freq, T]

    # 4) Mel projection
    mel = _MEL_FB @ spec                               # [60, T]

    # 5) log + mean-norm
    logmel = np.log(mel + _EPS, dtype=np.float16)
    logmel -= logmel.mean(axis=1, keepdims=True)

    # 6) pad / crop to 200 frames
    T = logmel.shape[1]
    if T < _TARGET_T:
        logmel = np.pad(logmel, ((0, 0), (0, _TARGET_T - T)), mode="constant")
        print(f"[INFO] Padding log_mel {T} → {_TARGET_T} frames")
    elif T > _TARGET_T:
        start = (T - _TARGET_T) // 2
        logmel = logmel[:, start:start + _TARGET_T]
        print(f"[INFO] Cropping log_mel {T} → {_TARGET_T} frames")

    return logmel[np.newaxis, np.newaxis, :, :].astype(np.float16)


# ───────────────────────── Embedding & similarity ──────────────────────────
def extract_embedding(rknn: RKNN, wav_path: str) -> np.ndarray:
    wave, sr = sf.read(wav_path, always_2d=False)
    if wave.ndim > 1:                                  # stereo → mono
        wave = wave.mean(axis=1)

    if sr != _SR:
        print(f"[INFO] Resampling {sr} Hz → {_SR} Hz")
        wave = resample(wave, int(len(wave) * _SR / sr))

    logmel_nchw = compute_logmel(wave)                 # [1,1,60,200]
    
    emb = rknn.inference(inputs=[logmel_nchw] , data_format='nchw' )[0]
    return emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + _EPS)


# ─────────────────────────────── main() ────────────────────────────────────
def main(model_path: str, wav1: str, wav2: str, target: str = "rk3588"):
    print(f"[1/4] Loading RKNN model: {model_path}")
    rk = RKNN()
    if rk.load_rknn(model_path) != 0:
        raise RuntimeError("❌ Failed to load RKNN model")

    print(f"[2/4] Initialising runtime for target: {target}")
    if rk.init_runtime(target=target) != 0:
        raise RuntimeError("❌ Failed to initialise RKNN runtime")

    print(f"[3/4] Extracting embeddings …")
    emb1 = extract_embedding(rk, wav1)
    emb2 = extract_embedding(rk, wav2)
    rk.release()

    print("[4/4] Computing cosine similarity …")
    sim = cosine_similarity(emb1, emb2)
    print(f"\n✅ Cosine Similarity : {sim:.4f}")
    print(f"🔎 Distance (1 - sim): {1 - sim:.4f}")


# ───────────────────────────── CLI wrapper ────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    model_path = sys.argv[1]
    wav1       = sys.argv[2]
    wav2       = sys.argv[3]
    target     = sys.argv[4] if len(sys.argv) > 4 else "rk3588"

    main(model_path, wav1, wav2, target)
