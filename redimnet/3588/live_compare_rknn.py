import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import get_window
from scipy.fftpack import fft
from rknn.api import RKNN


import sounddevice as sd
sd.default.device = (4, None)  
print(sd.query_devices())

# ------------------- Audio Preprocessing -------------------

def preemphasis(signal, alpha=0.97):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

def hz_to_mel(hz): return 2595 * np.log10(1 + hz / 700.0)
def mel_to_hz(mel): return 700 * (10**(mel / 2595.0) - 1)

def mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        start, center, end = bin_points[i - 1], bin_points[i], bin_points[i + 1]
        for j in range(start, center): fb[i - 1, j] = (j - start) / (center - start)
        for j in range(center, end): fb[i - 1, j] = (end - j) / (end - center)
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
    return pad_or_crop(log_mel, target_frames)[np.newaxis, np.newaxis, :, :]

def pad_or_crop(logmel, target_frames):
    n_mels, frames = logmel.shape
    if frames < target_frames:
        pad = np.zeros((n_mels, target_frames - frames))
        return np.concatenate((logmel, pad), axis=1)
    elif frames > target_frames:
        start = (frames - target_frames) // 2
        return logmel[:, start:start + target_frames]
    return logmel


# ------------------- Embedding & Similarity -------------------

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    v1 = emb1.flatten()
    v2 = emb2.flatten()
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

def extract_embedding(rknn: RKNN, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
    logmel_input = compute_logmel(waveform, sr=sr)
    return rknn.inference(inputs=[logmel_input.astype(np.float32)])[0]


# ------------------- Main Microphone Loop -------------------

def listen_and_compare(rknn_path, ref_wav_path, target='rk3588', chunk_secs=2, sr=16000):
    print(f"🧠 Loading model: {rknn_path}")
    rknn = RKNN()
    if rknn.load_rknn(rknn_path) != 0:
        raise RuntimeError("load_rknn failed")
    if rknn.init_runtime(target=target) != 0:
        raise RuntimeError("init_runtime failed")

    print(f"🎧 Loading reference audio: {ref_wav_path}")
    ref_wave, ref_sr = sf.read(ref_wav_path)
    if ref_wave.ndim > 1:
        ref_wave = ref_wave.mean(axis=1)
    ref_emb = extract_embedding(rknn, ref_wave, ref_sr)

    print("🎙️ Starting live microphone input...")
    duration = chunk_secs
    frames_per_chunk = int(sr * duration)

    try:
        while True:
            print("\n🔴 Speak now...")
            recording = sd.rec(frames_per_chunk, samplerate=sr, channels=1, dtype='float32')
            sd.wait()
            mic_wave = recording[:, 0]

            mic_emb = extract_embedding(rknn, mic_wave, sr)
            sim = cosine_similarity(ref_emb, mic_emb)

            print(f"🧭 Cosine similarity: {sim:.4f}   (distance: {1 - sim:.4f})")
    except KeyboardInterrupt:
        print("🛑 Stopped.")
        rknn.release()


# ------------------- Entry Point -------------------

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python live_compare_rknn.py model.rknn ref.wav [target_platform]")
        sys.exit(1)
    rknn_path = sys.argv[1]
    ref_wav = sys.argv[2]
    target = sys.argv[3] if len(sys.argv) > 3 else 'rk3588'
    listen_and_compare(rknn_path, ref_wav, target,1)

