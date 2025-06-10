import numpy as np
import soundfile as sf
from scipy.signal import get_window
from scipy.fftpack import fft
from rknn.api import RKNN


def preemphasis(signal, alpha=0.97):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700.0)

def mel_to_hz(mel):
    return 700 * (10**(mel / 2595.0) - 1)


def mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        start, center, end = bin_points[i - 1], bin_points[i], bin_points[i + 1]
        for j in range(start, center):
            fb[i - 1, j] = (j - start) / (center - start)
        for j in range(center, end):
            fb[i - 1, j] = (end - j) / (end - center)
    return fb


def compute_logmel(waveform, sr=16000, n_fft=512, hop_length=160, n_mels=60,
                   fmin=20.0, fmax=8000.0, target_frames=200):
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    waveform = preemphasis(waveform)

    window = get_window('hann', n_fft, fftbins=True)
    frames = []
    for start in range(0, len(waveform) - n_fft + 1, hop_length):
        frame = waveform[start:start + n_fft] * window
        spec = np.abs(fft(frame)[:n_fft // 2 + 1]) ** 2
        frames.append(spec)

    spec = np.stack(frames, axis=1)
    mel_fb = mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
    mel_spec = np.dot(mel_fb, spec)

    log_mel = np.log(mel_spec + 1e-6)
    log_mel = pad_or_crop(log_mel, target_frames)

    return log_mel[np.newaxis, np.newaxis, :, :]  # [1, 1, 60, 200]


def pad_or_crop(logmel, target_frames):
    n_mels, frames = logmel.shape
    if frames < target_frames:
        pad = np.zeros((n_mels, target_frames - frames))
        logmel = np.concatenate((logmel, pad), axis=1)
    elif frames > target_frames:
        start = (frames - target_frames) // 2
        logmel = logmel[:, start:start + target_frames]
    return logmel


def cosine_similarity_numpys(emb1: np.ndarray, emb2: np.ndarray) -> float:
    v1 = emb1.flatten()
    v2 = emb2.flatten()
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot / (norm1 * norm2 + 1e-8)


def extract_embedding(rknn: RKNN, wav_path: str) -> np.ndarray:
    waveform, sr = sf.read(wav_path)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    logmel_input = compute_logmel(waveform, sr=sr)
    output = rknn.inference(inputs=[logmel_input.astype(np.float32)])
    return output[0]


def main(model_path, wav1, wav2, target='rk3588'):
    print("Loading RKNN model...")
    rknn = RKNN()
    if rknn.load_rknn(model_path) != 0:
        raise RuntimeError("Failed to load RKNN model")

    print(f"Initializing RKNN runtime for target '{target}'...")
    if rknn.init_runtime(target=target) != 0:
        raise RuntimeError("Failed to init runtime")

    print(f"Extracting embeddings for:\n  {wav1}\n  {wav2}")
    emb1 = extract_embedding(rknn, wav1)
    emb2 = extract_embedding(rknn, wav2)

    rknn.release()

    sim = cosine_similarity_numpys(emb1, emb2)
    print(f"\n✅ Cosine similarity: {sim:.4f}")
    print(f"🔎 Distance (1 - sim): {1 - sim:.4f}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        print("Usage: python verify_voice_rknn.py model.rknn audio1.wav audio2.wav [target]")
        sys.exit(1)
    model = sys.argv[1]
    wav1 = sys.argv[2]
    wav2 = sys.argv[3]
    target = sys.argv[4] if len(sys.argv) > 4 else 'rk3588'
    main(model, wav1, wav2, target)

