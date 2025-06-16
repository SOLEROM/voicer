import numpy as np
import soundfile as sf
from scipy.signal import get_window, resample
from scipy.fftpack import fft
from rknn.api import RKNN

import os
os.environ['RKNN_LOG_LEVEL'] = '3'
# ========================== DSP HELPERS ==========================

def preemphasis(signal, alpha=0.97):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700.0)

def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)

def mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        start, center, end = bin_points[i - 1], bin_points[i], bin_points[i + 1]
        end = min(end, fb.shape[1])
        for j in range(start, center):
            fb[i - 1, j] = (j - start) / max(center - start, 1)
        for j in range(center, end):
            fb[i - 1, j] = (end - j) / max(end - center, 1)
    return fb

# ========================== LOG-MEL ==========================

def pad_or_crop(logmel, target_frames=200):
    n_mels, frames = logmel.shape
    if frames < target_frames:
        pad = np.zeros((n_mels, target_frames - frames))
        logmel = np.concatenate((logmel, pad), axis=1)
        print(f"Padding logmel from {frames} to {target_frames}")
    elif frames > target_frames:
        start = (frames - target_frames) // 2
        logmel = logmel[:, start:start + target_frames]
        print(f"Cropping logmel from {frames} to {target_frames}")
    return logmel

def compute_logmel(waveform, sr=16000, n_fft=512, hop_length=160, n_mels=60,
                   fmin=20.0, fmax=8000.0, target_frames=200, preemphasis_alpha=0.97):
    waveform = waveform.astype(np.float32)
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    waveform = preemphasis(waveform, alpha=preemphasis_alpha)

    window = get_window('hann', n_fft, fftbins=True)
    frames = []
    for start in range(0, len(waveform) - n_fft + 1, hop_length):
        frame = waveform[start:start + n_fft] * window
        spec = np.abs(fft(frame)[:n_fft // 2 + 1]) ** 2
        frames.append(spec)

    if not frames:
        raise ValueError("Audio too short for even one FFT frame")

    spec = np.stack(frames, axis=1)
    mel_fb = mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
    mel_spec = np.dot(mel_fb, spec)
    mel_spec = np.maximum(mel_spec, 1e-6)

    logmel = np.log(mel_spec)
    logmel = pad_or_crop(logmel, target_frames)

    # Standardization (identity for now)
    logmel = (logmel - 0.0) / 1.0

    print(f"[DEBUG] logmel shape = {logmel.shape}")
    return logmel[np.newaxis, np.newaxis, :, :]  # NCHW: [1, 1, n_mels, frames]

# ========================== INFERENCE ==========================

def run_inference_rknn(rknn_path, wav_path):
    print("[1/4] Loading RKNN model...")
    rknn = RKNN()
    if rknn.load_rknn(rknn_path) != 0:
        raise RuntimeError("❌ Failed to load RKNN")

    print("[2/4] Initializing runtime...")
    if rknn.init_runtime(target='rk3588') != 0:
        raise RuntimeError("❌ Failed to init runtime")

    print("[3/4] Preprocessing audio...")
    waveform, sr = sf.read(wav_path)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    target_sr = 16000
    if sr != target_sr:
        print(f"[INFO] Resampling from {sr} Hz to {target_sr} Hz...")
        duration_sec = len(waveform) / sr
        target_len = int(duration_sec * target_sr)
        waveform = resample(waveform, target_len)
        sr = target_sr

    logmel_nchw = compute_logmel(waveform, sr=sr)  # [1, 1, 60, 200]
    logmel_nhwc = np.transpose(logmel_nchw, (0, 2, 3, 1))  # [1, 60, 200, 1]

    print("[4/4] Running inference...")
    input_tensor = logmel_nhwc.astype(np.float32)
    outputs = rknn.inference(inputs=[input_tensor])
    rknn.release()

    embedding = outputs[0]

    print("\n[DEBUG] RKNN Output Stats:")
    print(f"Shape: {embedding.shape}")
    print(f"Min: {np.min(embedding):.4f}, Max: {np.max(embedding):.4f}")
    if np.isnan(embedding).any():
        print("⚠️  WARNING: NaN detected in output")
    if np.isinf(embedding).any():
        print("⚠️  WARNING: Inf detected in output")

    print("\n✅ Embedding vector preview:")
    print(embedding[0][:10])  # preview

    return embedding

# ========================== ENTRY ==========================

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python inference_rknn.py model.rknn audio.wav")
        sys.exit(1)
    run_inference_rknn(sys.argv[1], sys.argv[2])
