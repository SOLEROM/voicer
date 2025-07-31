#!/usr/bin/env python3
"""
build_ref_embedding.py – create an averaged speaker embedding
                         from a long recording using a ReDimNet-NoMel RKNN.

Usage
-----
python build_ref_embedding.py  model.rknn  long_recording.wav  out_embed.pt  [chunk_secs]  [rk3588]

Default chunk length = 2 s (134 frames with the IDRnD parameters).

The output file is a Torch tensor (CPU, float32) saved with torch.save().
"""

# ───────────────────────── Imports ─────────────────────────
import os
import sys
import numpy as np
import soundfile as sf
from scipy.signal import get_window, resample
from numpy.fft    import rfft
from rknn.api     import RKNN
import torch                         # only for torch.save()

# Optional: keep RKNN quiet
os.environ["RKNN_LOG_LEVEL"] = "3"

# ─────────────────── Front-end constants (IDRnD) ───────────────────
_PREEMPH  = 0.97
_SR       = 16_000
_N_FFT    = 512
_WIN_LEN  = 400
_HOP      = 240
_N_MELS   = 60
_F_MIN    = 20.0
_F_MAX    = 7_600.0
_TARGET_T = 134                     # frames for a 2-s clip
_EPS      = 1e-6

# ─────────────────── DSP helpers ───────────────────
def preemphasis(wave, alpha=_PREEMPH):
    out = wave.copy()
    out[1:] -= alpha * wave[:-1]
    return out


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asanyarray(hz) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0**(np.asanyarray(mel) / 2595.0) - 1.0)


def mel_filterbank(sr=_SR, n_fft=_N_FFT, n_mels=_N_MELS,
                   f_min=_F_MIN, f_max=_F_MAX):
    mel_pts = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_pts  = mel_to_hz(mel_pts)
    bins    = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float16)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        right = min(right, fb.shape[1] - 1)
        if center > left:
            fb[i - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            fb[i - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    return fb


_MEL_FB = mel_filterbank()
_WINDOW = get_window("hamming", _WIN_LEN, fftbins=True).astype(np.float16)
_WINDOW = np.pad(_WINDOW, (0, _N_FFT - _WIN_LEN))      # zero-pad to n_fft


def waveform_to_logmel(wave):
    """Wave → log-Mel [1,1,60,134]   (float16 NCHW)."""
    wave = preemphasis(wave).astype(np.float16)
    pad  = _N_FFT // 2
    wave = np.pad(wave, (pad, pad), mode="reflect")

    frames = []
    for start in range(0, len(wave) - _N_FFT + 1, _HOP):
        frame = wave[start:start + _N_FFT] * _WINDOW
        spec  = np.abs(rfft(frame, n=_N_FFT)) ** 2
        frames.append(spec)
    if not frames:
        raise RuntimeError("audio chunk too short")

    spec = np.stack(frames, axis=1, dtype=np.float16)   # [freq, T]
    mel  = _MEL_FB @ spec                               # [60, T]
    logmel = np.log(mel + _EPS, dtype=np.float16)
    logmel -= logmel.mean(axis=1, keepdims=True)

    if logmel.shape[1] != _TARGET_T:                    # pad or crop
        T = logmel.shape[1]
        if T < _TARGET_T:
            logmel = np.pad(logmel, ((0, 0), (0, _TARGET_T - T)))
        else:
            start = (T - _TARGET_T) // 2
            logmel = logmel[:, start:start + _TARGET_T]

    return logmel[np.newaxis, np.newaxis].astype(np.float16)      # [1,1,60,134]


# ─────────────────── Embedding extraction ───────────────────
def extract_embedding(rknn, wave):
    logmel = waveform_to_logmel(wave)
    return rknn.inference(inputs=[logmel], data_format="nchw")[0]   # (1, 192) float32


# ─────────────────── Main pipeline ───────────────────
def build_reference(model_path, wav_path, out_path,
                    chunk_secs=2, target="rk3588"):

    # 1. Initialise RKNN
    rk = RKNN()
    if rk.load_rknn(model_path) != 0:
        sys.exit("load_rknn failed")
    if rk.init_runtime(target=target) != 0:
        sys.exit("init_runtime failed")

    # 2. Read audio
    wave, sr = sf.read(wav_path, always_2d=False)
    if wave.ndim > 1:      # stereo → mono
        wave = wave.mean(axis=1)
    if sr != _SR:
        wave = resample(wave, int(len(wave) / sr * _SR))
        sr = _SR

    chunk_len = int(chunk_secs * sr)
    n_chunks  = len(wave) // chunk_len
    if n_chunks == 0:
        sys.exit("Recording shorter than one chunk")

    print(f"[INFO] Processing {n_chunks} chunks of {chunk_secs:.1f} s …")
    embeds = []
    for i in range(n_chunks):
        chunk = wave[i * chunk_len : (i + 1) * chunk_len]
        emb   = extract_embedding(rk, chunk)[0]          # (192,)
        embeds.append(emb)

    rk.release()

    embeds = np.vstack(embeds)               # [N, 192]
    # Optional: length-normalise each embedding before averaging
    embeds /= np.linalg.norm(embeds, axis=1, keepdims=True) + _EPS
    avg_emb = embeds.mean(axis=0)
    avg_emb = avg_emb / (np.linalg.norm(avg_emb) + _EPS)  # final L2 norm

    torch.save(torch.from_numpy(avg_emb.astype(np.float32)), out_path)
    print(f"[✓] Saved averaged embedding to {out_path}")
    return avg_emb


# ─────────────────── CLI entry-point ───────────────────
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    model_file   = sys.argv[1]
    wav_file     = sys.argv[2]
    out_file     = sys.argv[3]
    chunk_secs   = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    target_hw    = sys.argv[5] if len(sys.argv) > 5 else "rk3588"

    build_reference(model_file, wav_file, out_file, chunk_secs, target_hw)
