#!/usr/bin/env python3
"""
inference_rknn.py – Rockchip RKNN inference for ReDimNetNoMel checkpoints
(NumPy/SciPy implementation of the original PyTorch front-end)

Usage
-----
python inference_rknn.py  model.rknn  audio.wav
"""

# ----------------------------------------------------------------------
#  Imports (no PyTorch, no torchaudio)
# ----------------------------------------------------------------------
import os
import sys
import numpy as np
import soundfile as sf
from scipy.signal import get_window, resample
from numpy.fft    import rfft
from rknn.api     import RKNN

# ----------------------------------------------------------------------
#  RKNN log level (3 = warnings and up)
# ----------------------------------------------------------------------
os.environ["RKNN_LOG_LEVEL"] = "3"

# ----------------------------------------------------------------------
#  Front-end constants  (IDRnD defaults)
# ------------------------------------------------------------------
_PREEMPH  = 0.97
_SR       = 16_000
_N_FFT    = 512
_WIN_LEN  = 400
_HOP      = 240
_N_MELS   = 60
_F_MIN    = 20.0
_F_MAX    = 7_600.0
_TARGET_T = 200
_EPS      = 1e-6

# ------------------------------------------------------------------
#  DSP helpers
# ------------------------------------------------------------------
def preemphasis(wave: np.ndarray, alpha: float = _PREEMPH) -> np.ndarray:
    """ y[n] = x[n] − α·x[n−1]   (first sample unchanged) """
    out = wave.copy()
    out[1:] -= alpha * wave[:-1]
    return out


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    """HTK mel conversion (same as torchaudio MelScale default)"""
    return 2595.0 * np.log10(1.0 + np.asanyarray(hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asanyarray(mel) / 2595.0) - 1.0)


def mel_filterbank(sr: int              = _SR,
                   n_fft: int           = _N_FFT,
                   n_mels: int          = _N_MELS,
                   f_min: float         = _F_MIN,
                   f_max: float         = _F_MAX) -> np.ndarray:
    """
    Return triangular filterbank of shape (n_mels, n_fft//2 + 1)
    matching torchaudio’s default (HTK formula).
    """
    mel_pts = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_pts  = mel_to_hz(mel_pts)
    bins    = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        right = min(right, fb.shape[1] - 1)              # safety clip

        # rising slope
        if center > left:
            fb[i - 1, left:center] = (
                np.arange(left, center) - left) / (center - left)
        # falling slope
        if right > center:
            fb[i - 1, center:right] = (
                right - np.arange(center, right)) / (right - center)
    return fb


# Pre-compute once – reuse for every clip
_MEL_FB  = mel_filterbank()
_WINDOW  = get_window("hamming", _WIN_LEN, fftbins=True).astype(np.float32)
_WINDOW  = np.pad(_WINDOW, (0, _N_FFT - _WIN_LEN))       # zero-pad to n_fft


# ------------------------------------------------------------------
#  Waveform → log-Mel (NumPy implementation of your PyTorch logic)
# ------------------------------------------------------------------
def waveform_to_logmel(wave: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    wave : np.ndarray  shape (N,) – mono 16-kHz waveform

    Returns
    -------
    logmel : np.ndarray  shape (1, 1, 60, 200)  (NCHW float32)
    """
    # 1) pre-emphasis
    wave = preemphasis(wave, _PREEMPH).astype(np.float32)

    # 2) reflection pad to mimic torch.stft(center=True)
    pad = _N_FFT // 2
    wave_padded = np.pad(wave, (pad, pad), mode="reflect")

    # 3) frame-by-frame STFT power
    frames = []
    for start in range(0, len(wave_padded) - _N_FFT + 1, _HOP):
        frame = wave_padded[start:start + _N_FFT] * _WINDOW
        spec  = np.abs(rfft(frame, n=_N_FFT)) ** 2
        frames.append(spec)

    if not frames:
        raise RuntimeError("❌ Audio too short for one FFT frame")

    spec = np.stack(frames, axis=1, dtype=np.float32)            # [freq, T]

    # 4) Mel projection
    mel = _MEL_FB @ spec                                           # [60, T]

    # 5) log (natural)
    logmel = np.log(mel + _EPS, dtype=np.float32)

    # 6) mean normalisation (per mel bin)
    logmel -= logmel.mean(axis=1, keepdims=True)

    # 7) pad / crop to exactly 200 frames
    T = logmel.shape[1]
    if T < _TARGET_T:
        logmel = np.pad(logmel, ((0, 0), (0, _TARGET_T - T)), mode="constant")
        print(f"[INFO] Padding log_mel from {T} → {_TARGET_T} frames")
    elif T > _TARGET_T:
        start = (T - _TARGET_T) // 2
        logmel = logmel[:, start:start + _TARGET_T]
        print(f"[INFO] Cropping log_mel from {T} → {_TARGET_T} frames")

    # 8) return [1, 1, 60, 200] (NCHW)
    return logmel[np.newaxis, np.newaxis, :, :].astype(np.float32)


# ------------------------------------------------------------------
#  RKNN inference wrapper
# ------------------------------------------------------------------
def run_inference_rknn(rknn_path: str, wav_path: str):
    print("[1/4] Loading RKNN model …")
    rk = RKNN()
    if rk.load_rknn(rknn_path) != 0:
        raise RuntimeError("❌ Failed to load RKNN file")

    print("[2/4] Initialising runtime …")
    if rk.init_runtime(target="rk3588") != 0:
        raise RuntimeError("❌ Failed to initialise RKNN runtime")

    # ------------------------------------------------------------------
    #  Audio I/O  → log-Mel
    # ------------------------------------------------------------------
    print("[3/4] Pre-processing audio …")
    wave, sr = sf.read(wav_path, always_2d=False)    # mono or stereo
    if wave.ndim > 1:                                # stereo → mono
        wave = wave.mean(axis=1)

    if sr != _SR:
        print(f"[INFO] Resampling {sr} → {_SR} Hz")
        duration = len(wave) / sr
        wave = resample(wave, int(duration * _SR))
        sr = _SR

    logmel_nchw = waveform_to_logmel(wave)           # [1,1,60,200]
    logmel_nhwc = np.transpose(logmel_nchw, (0, 2, 3, 1))  # → NHWC

    # ------------------------------------------------------------------
    #  Inference
    # ------------------------------------------------------------------
    print("[4/4] Running inference …")
    outputs = rk.inference(inputs=[logmel_nhwc])
    rk.release()

    emb = outputs[0]
    print(f"\n[DEBUG] Embedding stats – shape {emb.shape}  "
          f"min={emb.min():.4f}  max={emb.max():.4f}")
    print("Sample values:", emb[0, :10])

    return emb


# ------------------------------------------------------------------
#  CLI entry-point
# ------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    run_inference_rknn(sys.argv[1], sys.argv[2])
