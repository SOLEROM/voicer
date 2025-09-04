#!/usr/bin/env python3
"""
live_compare_rknn.py – Real-time microphone verification with a ReDimNet-NoMel
RKNN model.

The reference can be **either**:
  • an audio file (wav/flac/…) – its embedding is computed at start time, or
  • a Torch-saved embedding tensor (.pt/.torch) or a .npy file.

Usage
-----
python live_compare_rknn.py  model.rknn  reference.{wav|pt|torch|npy}
                             [rk3588]  [chunk_secs]  [modelB2|modelB0]  [debug]

Notes
-----
- Model variant controls number of Mel bands:
    * modelB2 → 72 mels
    * modelB0 → 60 mels
- Debug mode can be enabled by:
    * trailing CLI arg "debug" (any case), or
    * environment variable DEBUG=1 / true / yes
  When debug is ON:
    * RKNN runtime uses perf_debug=True
    * Per-inference timing is printed
"""

# ───────────────────────── Imports ─────────────────────────
import os
import sys
import time
import enum
import numpy as np
import sounddevice as sd
import soundfile  as sf
from functools import lru_cache
from scipy.signal import get_window, resample
from numpy.fft    import rfft
from rknn.api     import RKNN

try:
    import torch           # optional, only for torch.load()
except ImportError:
    torch = None

# os.environ['RKNN_LOG_LEVEL'] = '3'        # warnings and up only


# ─────────────────── Model variant enum ───────────────────
class ModelVariant(enum.Enum):
    modelB2 = 72
    modelB0 = 60

    @staticmethod
    def from_string(s: str) -> "ModelVariant":
        s = (s or "").strip().lower()
        if s in ("modelb2", "b2"):
            return ModelVariant.modelB2
        if s in ("modelb0", "b0"):
            return ModelVariant.modelB0
        return ModelVariant.modelB2   # default


# ─────────────────── MIC SRC    ───────────────────
def set_input_device(preferred_name="ReSpeaker", fallback_id=0, verbose=True):
    try:
        if verbose:
            print(f"🔍 Searching for input device containing: '{preferred_name}'")
        devices = sd.query_devices()
        input_devices = [(i, d) for i, d in enumerate(devices) if d['max_input_channels'] > 0]

        for idx, device in input_devices:
            if preferred_name.lower() in device['name'].lower():
                sd.default.device = (idx, None)
                if verbose:
                    print(f"✅ Using input device #{idx}: {device['name']}")
                break
        else:
            if verbose:
                print(f"⚠️ '{preferred_name}' not found. Falling back to device #{fallback_id}: {devices[fallback_id]['name']}")
            sd.default.device = (fallback_id, None)

    except Exception as e:
        print(f"❌ Error setting input device. Falling back to #{fallback_id}. Error: {e}")
        sd.default.device = (fallback_id, None)

    if verbose:
        print("🎙️ Final input device:", sd.query_devices(sd.default.device[0])['name'])


# ─────────────────── Front-end constants (IDRnD) ───────────────────
_PREEMPH  = 0.97
_SR       = 16_000
_N_FFT    = 512
_WIN_LEN  = 400
_HOP      = 240
# n_mels is chosen via ModelVariant
_F_MIN    = 20.0
_F_MAX    = 7_600.0
_TARGET_T = 134                            # frames used by ReDimNet-NoMel
_EPS      = 1e-6


# ─────────────────── DSP helpers ───────────────────
def preemphasis(wave: np.ndarray, alpha: float = _PREEMPH) -> np.ndarray:
    out = wave.copy()
    out[1:] -= alpha * wave[:-1]
    return out


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asanyarray(hz) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0**(np.asanyarray(mel) / 2595.0) - 1.0)


@lru_cache(maxsize=8)
def mel_filterbank_cached(sr=_SR, n_fft=_N_FFT, n_mels=72,
                          f_min=_F_MIN, f_max=_F_MAX) -> np.ndarray:
    """Triangular filterbank matching torchaudio defaults (HTK).
    Cached per (sr, n_fft, n_mels, f_min, f_max).
    """
    mel_pts = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_pts  = mel_to_hz(mel_pts)
    bins    = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float16)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        right = min(right, fb.shape[1] - 1)

        if center > left:                              # rising slope
            fb[i - 1, left:center] = (
                np.arange(left, center) - left) / (center - left)
        if right > center:                             # falling slope
            fb[i - 1, center:right] = (
                right - np.arange(center, right)) / (right - center)
    return fb


# Pre-compute static window once (independent of n_mels)
_WINDOW = get_window('hamming', _WIN_LEN, fftbins=True).astype(np.float16)
_WINDOW = np.pad(_WINDOW, (0, _N_FFT - _WIN_LEN))      # zero-pad to n_fft


def waveform_to_logmel(wave: np.ndarray,
                       n_mels: int,
                       target_frames=_TARGET_T) -> np.ndarray:
    """
    Convert mono 16-kHz waveform → log-Mel tensor [1,1,n_mels,target_frames].
    All maths in float16 to save RAM/BW like on RK NPU.
    """
    # 1) pre-emphasis
    wave = preemphasis(wave).astype(np.float16)

    # 2) reflection pad (torch.stft(center=True) equivalence)
    pad = _N_FFT // 2
    wave = np.pad(wave, (pad, pad), mode='reflect')

    # 3) STFT power spectrum for every frame
    frames = []
    for start in range(0, len(wave) - _N_FFT + 1, _HOP):
        frame = wave[start:start + _N_FFT] * _WINDOW
        spec  = np.abs(rfft(frame, n=_N_FFT)) ** 2      # power
        frames.append(spec)
    if not frames:
        raise RuntimeError('❌ Audio too short for one FFT frame')

    spec = np.stack(frames, axis=1, dtype=np.float16)   # [freq, T]

    # 4) Mel projection
    fb  = mel_filterbank_cached(_SR, _N_FFT, n_mels, _F_MIN, _F_MAX)
    mel = fb @ spec                                     # [n_mels, T]

    # 5) natural log + per-bin mean normalisation
    logmel = np.log(mel + _EPS, dtype=np.float16)
    logmel -= logmel.mean(axis=1, keepdims=True)

    # 6) pad / centre-crop to target_frames
    T = logmel.shape[1]
    if T < target_frames:
        logmel = np.pad(logmel, ((0, 0), (0, target_frames - T)),
                        mode='constant')
    elif T > target_frames:
        start = (T - target_frames) // 2
        logmel = logmel[:, start:start + target_frames]

    # 7) NCHW for ReDimNet-NoMel: [B, C, H, W] = [1,1,n_mels,T]
    return logmel[np.newaxis, np.newaxis].astype(np.float16)


# ─────────────────── Embedding helpers ───────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) /
                 (np.linalg.norm(a) * np.linalg.norm(b) + _EPS))


def extract_embedding(rknn: RKNN, wave: np.ndarray, sr: int,
                      n_mels: int,
                      debug: bool = False) -> np.ndarray:
    """Resample if needed, run front-end and RKNN inference, return embedding."""
    if sr != _SR:
        duration = len(wave) / sr
        wave = resample(wave, int(duration * _SR))
    logmel = waveform_to_logmel(wave, n_mels=n_mels)  # [1,1,n_mels,T]

    if debug:
        t0 = time.time()
        out = rknn.inference(inputs=[logmel], data_format='nchw')[0]
        t1 = time.time()
        print(f"Inference time: {(t1 - t0)*1000:.2f} ms")
    else:
        out = rknn.inference(inputs=[logmel], data_format='nchw')[0]

    return out.astype(np.float16)


def load_reference_embedding(path: str) -> np.ndarray:
    """
    Load a Torch-saved embedding (.pt/.torch) **or** a .npy file and
    return float16 NumPy array.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        ref = np.load(path).astype(np.float16)
    else:
        if torch is None:
            sys.exit('ERROR: torch is required to load Torch tensors (.pt/.torch).')
        ref = torch.load(path, map_location='cpu')
        if isinstance(ref, torch.Tensor):
            ref = ref.cpu().to(torch.float16).numpy()
        else:
            ref = np.asarray(ref, dtype=np.float16)
    return ref


def prepare_reference(rknn: RKNN, ref_path: str,
                      n_mels: int,
                      debug: bool = False) -> np.ndarray:
    """
    Determine whether `ref_path` is audio or an embedding file and return the embedding.
    """
    audio_exts = {'.wav', '.flac', '.ogg', '.mp3', '.m4a', '.aac'}
    ext = os.path.splitext(ref_path)[1].lower()
    if ext in audio_exts:
        if debug:
            print(f"🎧 Computing reference embedding from audio: {ref_path}")
        wave, sr = sf.read(ref_path, always_2d=False)
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        return extract_embedding(rknn, wave, sr, n_mels=n_mels, debug=debug)
    else:
        if debug:
            print(f"📦 Loading reference embedding from file: {ref_path}")
        return load_reference_embedding(ref_path)


# ─────────────────── Main loop ───────────────────
def listen_and_compare(rknn_path: str,
                       ref_path:   str,
                       target:     str  = 'rk3588',
                       chunk_s:    int  = 1,
                       variant:    ModelVariant = ModelVariant.modelB2,
                       debug:      bool = False):
    print(f'🧠 Loading RKNN model: {rknn_path}')
    rk = RKNN()
    if rk.load_rknn(rknn_path) != 0:
        sys.exit('load_rknn failed')

    # perf_debug only when debug mode is ON
    if rk.init_runtime(target=target, perf_debug=bool(debug)) != 0:
        sys.exit('init_runtime failed')

    n_mels = variant.value
    print(f'🎚️ Using model variant {variant.name} with n_mels={n_mels}')

    print(f'🔧 Preparing reference from: {ref_path}')
    ref_emb = prepare_reference(rk, ref_path, n_mels=n_mels, debug=debug)

    sd.default.samplerate = _SR        # record natively at 16 kHz
    sd.default.channels   = 1

    print('\n🎙️  Speak whenever you like – Ctrl-C to quit.')
    frames_per_chunk = int(_SR * chunk_s)
    try:
        while True:
            print('\n🔴 Recording…')
            rec = sd.rec(frames_per_chunk, dtype='float32')
            sd.wait()
            mic_wave = rec[:, 0]                      # 1-D mono
            mic_emb  = extract_embedding(rk, mic_wave, _SR, n_mels=n_mels, debug=debug)
            sim      = cosine_similarity(ref_emb, mic_emb)
            print(f'🧭 Cosine similarity: {sim:.4f}   (distance = {1-sim:.4f})')
    except KeyboardInterrupt:
        print('\n🛑 Stopped by user.')
    finally:
        rk.release()


# ─────────────────── CLI entry-point ───────────────────
def _env_debug_on() -> bool:
    v = os.environ.get("DEBUG", "").strip()
    return v in ("1", "true", "True", "YES", "yes")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    rknn_model = sys.argv[1]
    reference  = sys.argv[2]
    target_hw  = sys.argv[3] if len(sys.argv) > 3 else 'rk3588'
    chunk_secs = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    variant_arg = sys.argv[5] if len(sys.argv) > 5 else None
    debug_arg   = sys.argv[6] if len(sys.argv) > 6 else None

    model_variant = ModelVariant.from_string(variant_arg)
    debugMode = _env_debug_on() or (str(debug_arg).lower() == "debug")

    # device selection (quiet unless debug)
    set_input_device("ReSpeaker", fallback_id=4, verbose=debugMode)
    if debugMode:
        print(sd.query_devices())

    listen_and_compare(
        rknn_model,
        reference,
        target_hw,
        chunk_secs,
        variant=model_variant,
        debug=debugMode
    )
