#!/usr/bin/env python3
"""
compare_voice_to_ref.py – Verify one WAV file against a stored reference
embedding using a ReDimNetNoMel RKNN model.

Example
-------
python compare_voice_to_ref.py  model.rknn  probe.wav  ref_embed.pt  [rk3588]

Notes
-----
* The reference file **must** be a .pt created by `torch.save(tensor, path)`.
  Only CPU tensors are expected.
* PyTorch is imported **only** for torch.load; all DSP is pure NumPy/SciPy.
"""

# ───────────────────────── Imports ─────────────────────────
import os
import sys
import io
import numpy as np
import soundfile as sf
from scipy.signal import get_window, resample
from numpy.fft    import rfft
from rknn.api     import RKNN
import torch      # needed solely for torch.load()

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
_TARGET_T = 200
_EPS      = 1e-6

# ─────────────────── DSP helpers ───────────────────
def preemphasis(wave, alpha=_PREEMPH):
    out = wave.copy()
    out[1:] -= alpha * wave[:-1]
    return out

def hz_to_mel(h):      return 2595.0 * np.log10(1.0 + h / 700.0)
def mel_to_hz(m):      return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

def mel_filterbank(sr=_SR, n_fft=_N_FFT, n_mels=_N_MELS,
                   fmin=_F_MIN, fmax=_F_MAX):
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts  = mel_to_hz(mel_pts)
    bins    = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        r = min(r, fb.shape[1] - 1)
        if c > l:
            fb[i - 1, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c:
            fb[i - 1, c:r] = (r - np.arange(c, r)) / (r - c)
    return fb

# Pre-compute static pieces
_MEL_FB = mel_filterbank()
_WINDOW = np.pad(get_window("hamming", _WIN_LEN, fftbins=True)
                 .astype(np.float32), (0, _N_FFT - _WIN_LEN))

# ─────────────────── Wave → log-Mel ───────────────────
def compute_logmel(wave):
    wave = preemphasis(wave).astype(np.float32)
    pad  = _N_FFT // 2
    wave = np.pad(wave, (pad, pad), mode="reflect")

    frames = []
    for s in range(0, len(wave) - _N_FFT + 1, _HOP):
        frame = wave[s:s + _N_FFT] * _WINDOW
        spec  = np.abs(rfft(frame, n=_N_FFT)) ** 2
        frames.append(spec)
    if not frames:
        raise RuntimeError("❌ Audio too short for one FFT frame")

    spec = np.stack(frames, axis=1)
    mel  = _MEL_FB @ spec
    logm = np.log(mel + _EPS, dtype=np.float32)
    logm -= logm.mean(axis=1, keepdims=True)

    T = logm.shape[1]
    if T < _TARGET_T:
        logm = np.pad(logm, ((0, 0), (0, _TARGET_T - T)), mode="constant")
        print(f"[INFO] Padding log_mel {T} → {_TARGET_T} frames")
    elif T > _TARGET_T:
        st = (T - _TARGET_T) // 2
        logm = logm[:, st:st + _TARGET_T]
        print(f"[INFO] Cropping log_mel {T} → {_TARGET_T} frames")

    return logm[np.newaxis, np.newaxis, :, :].astype(np.float32)

# ─────────────────── RKNN embedding ───────────────────
def extract_embedding(rknn: RKNN, wav_path):
    wave, sr = sf.read(wav_path, always_2d=False)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if sr != _SR:
        print(f"[INFO] Resampling {sr} Hz → {_SR} Hz")
        wave = resample(wave, int(len(wave) * _SR / sr))

    logmel = compute_logmel(wave)                       # NCHW
    logmel_nhwc = np.transpose(logmel, (0, 2, 3, 1))   # NHWC
    return rknn.inference(inputs=[logmel_nhwc])[0]

# ─────────────────── reference loader ───────────────────
def load_reference_embedding(path):
    """
    Loads a .pt saved with torch.save(). Converts to NumPy array.
    """
    try:
        ref = torch.load(path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load reference embedding: {e}")

    if isinstance(ref, torch.Tensor):
        return ref.cpu().numpy()
    elif isinstance(ref, (np.ndarray, list, tuple)):
        return np.asarray(ref, dtype=np.float32)
    else:
        raise TypeError("Unsupported reference format (expect tensor/array)")

# ─────────────────── cosine sim ───────────────────
def cosine_similarity(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + _EPS))

# ─────────────────── main ───────────────────
def main(model, wav_path, ref_path, target="rk3588"):
    print(f"[1/4] Loading RKNN model: {model}")
    rk = RKNN()
    if rk.load_rknn(model) != 0:
        raise RuntimeError("❌ Failed to load RKNN model")

    print(f"[2/4] Initialising runtime for target: {target}")
    if rk.init_runtime(target=target) != 0:
        raise RuntimeError("❌ Failed to initialise RKNN runtime")

    print(f"[3/4] Extracting embedding from {wav_path} …")
    probe_emb = extract_embedding(rk, wav_path)
    rk.release()

    print(f"[4/4] Loading reference embedding from {ref_path} …")
    ref_emb = load_reference_embedding(ref_path)

    sim = cosine_similarity(probe_emb, ref_emb)
    print(f"\n✅ Cosine Similarity : {sim:.4f}")
    print(f"🔎 Distance (1 - sim): {1 - sim:.4f}")

# ─────────────────── CLI ───────────────────
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    model_path  = sys.argv[1]
    wav_path    = sys.argv[2]
    ref_pt_path = sys.argv[3]
    target_chip = sys.argv[4] if len(sys.argv) > 4 else "rk3588"

    main(model_path, wav_path, ref_pt_path, target_chip)
