#!/usr/bin/env python3
"""
Generate INT8-calibration samples for ReDimNetNoMel.
Each .npy file contains:  shape [1, 1, 60, 200]  (float32)
and is written to calib_npy/sample_<idx>.npy
"""
import argparse, os, librosa, numpy as np
from pathlib import Path

# ---------- parameters (match your model!) ----------
SAMPLE_RATE   = 16_000          # Hz
N_MELS        = 60
N_FFT         = 512
HOP_LENGTH    = 160
TARGET_FRAMES = 200             # time axis after crop/pad
# ----------------------------------------------------

def pad_or_crop(mel: np.ndarray, T=TARGET_FRAMES):
    """mel: [n_mels, time]  -> [n_mels, T]"""
    if mel.shape[1] < T:                       # pad right
        pad = np.zeros((N_MELS, T - mel.shape[1]), mel.dtype)
        mel = np.concatenate([mel, pad], axis=1)
    elif mel.shape[1] > T:                     # centre-crop
        start = (mel.shape[1] - T) // 2
        mel = mel[:, start:start+T]
    return mel

def wav_to_logmel(path: Path):
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    mel = librosa.feature.melspectrogram(
        y, sr=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS,
        fmin=20, fmax=7600, power=2.0)
    logmel = librosa.power_to_db(mel, ref=1.0)           # log10(mel)
    logmel = (logmel - logmel.mean()) / (logmel.std()+1e-8)  # zero-mean, unit-std
    return pad_or_crop(logmel).astype(np.float32)        # [60, 200]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--out-dir", default="calib_npy")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    wavs = sorted(Path(args.wav_dir).glob("*.wav"))
    if not wavs:
        raise SystemExit("No *.wav files found in {}".format(args.wav_dir))

    for idx, w in enumerate(wavs):
        x = wav_to_logmel(w)                  # [60,200]
        x = x[None, None, ...]                # [1,1,60,200]
        np.save(out_dir / f"sample_{idx}.npy", x)
        print("✓", w.name, "→", f"sample_{idx}.npy")

    # write dataset.txt
    with open("dataset.txt", "w") as f:
        for idx in range(len(wavs)):
            f.write(f"{out_dir}/sample_{idx}.npy\n")

if __name__ == "__main__":
    main()
