#!/usr/bin/env python3
"""
compare_voice_to_ref_fp16.py
----------------------------
Verify one WAV file against a stored reference embedding (saved with
torch.save) using a ReDimNet-NoMel RKNN model – **all data passed as FP-16**.

Example
-------
python compare_voice_to_ref_fp16.py \
       model.rknn  probe.wav  ref_embed.pt  [rk3588]
"""

# ───────────────────────── Imports ─────────────────────────
import os
import sys
import numpy as np
import soundfile as sf
from scipy.signal import get_window, resample
from numpy.fft    import rfft
from rknn.api     import RKNN
import torch                         # only for torch.load()

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

def hz_to_mel(h):  return 2595.0 * np.log10(1.0 + h / 700.0)
def mel_to_hz(m):  return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

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

# Pre-compute static pieces (float32)
_MEL_FB = mel_filterbank()
_WINDOW = np.pad(get_window("hamming", _WIN_LEN, fftbins=True)
                 .astype(np.float32), (0, _N_FFT - _WIN_LEN))

# ─────────────────── Wave → log-Mel (FP-16 output) ───────────────────
def compute_logmel(wave: np.ndarray) -> np.ndarray:
    """
    Return NCHW tensor as **float16**  [1,1,60,200].
    """
    wave = preemphasis(wave).astype(np.float32)          # keep DSP in 32-bit
    wave = np.pad(wave, (_N_FFT // 2, _N_FFT // 2), mode="reflect")

    frames = []
    for s in range(0, len(wave) - _N_FFT + 1, _HOP):
        frame = wave[s:s + _N_FFT] * _WINDOW
        frames.append(np.abs(rfft(frame, n=_N_FFT)) ** 2)

    if not frames:
        raise RuntimeError("❌ Audio too short for one FFT frame")

    spec = np.stack(frames, axis=1)
    mel  = _MEL_FB @ spec
    logm = np.log(mel + _EPS, dtype=np.float32)
    logm -= logm.mean(axis=1, keepdims=True)

    T = logm.shape[1]
    if T < _TARGET_T:
        logm = np.pad(logm, ((0, 0), (0, _TARGET_T - T)), mode="constant")
    elif T > _TARGET_T:
        st = (T - _TARGET_T) // 2
        logm = logm[:, st:st + _TARGET_T]

    return logm[np.newaxis, np.newaxis, :, :].astype(np.float16)

# ─────────────────── RKNN embedding (FP-16 I/O) ───────────────────
def extract_embedding(rknn: RKNN, wav_path: str) -> np.ndarray:
    wave, sr = sf.read(wav_path, always_2d=False)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if sr != _SR:
        wave = resample(wave, int(len(wave) * _SR / sr))

    logmel16 = compute_logmel(wave)                      # [1,1,60,200] fp16
    logmel16_nhwc = np.transpose(logmel16, (0, 2, 3, 1))
    return rknn.inference(inputs=[logmel16_nhwc])[0].astype(np.float16, copy=False)

# ─────────────────── reference loader (fp16) ───────────────────
def load_reference_embedding(path: str) -> np.ndarray:
    ref = torch.load(path, map_location="cpu")
    if isinstance(ref, torch.Tensor):
        ref = ref.cpu().to(torch.float16).numpy()
    else:
        ref = np.asarray(ref, dtype=np.float16)
    return ref

# ─────────────────── cosine similarity (fp16 → fp32) ───────────────────
def cosine_similarity(a16: np.ndarray, b16: np.ndarray) -> float:
    a = a16.astype(np.float32, copy=False).flatten()
    b = b16.astype(np.float32, copy=False).flatten()
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

    print(f"[3/4] Extracting probe embedding …")
    probe_emb16 = extract_embedding(rk, wav_path)
    rk.release()

    print(f"[4/4] Loading reference embedding …")
    ref_emb16 = load_reference_embedding(ref_path)

    sim = cosine_similarity(probe_emb16, ref_emb16)
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
