import onnx
import onnxruntime as ort
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np


ONNX_MODEL_PATH = "ReDimNet_no_mel.onnx"


def load_and_verify_model(onnx_path: str):
    """Load and verify ONNX model once"""
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("ONNX model is valid!")
    session = ort.InferenceSession(onnx_path)
    return session


def waveform_to_logmel(
    waveform: torch.Tensor,
    sample_rate=16000,
    n_fft=512,
    hop_length=160,
    n_mels=60,
    f_min=20.0,
    f_max=8000.0,
    preemphasis_alpha=0.97
):
    waveform = waveform / (waveform.abs().max() + 1e-8)

    shifted = torch.roll(waveform, shifts=1, dims=1)
    waveform_preemph = waveform - preemphasis_alpha * shifted
    waveform_preemph[:, 0] = waveform[:, 0]

    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=2.0,
        center=False
    )
    mel_spec = mel_transform(waveform_preemph)
    log_mel = torch.log(mel_spec + 1e-6)
    return log_mel


def run_inference_onnx(session: ort.InferenceSession, wav_path: str):
    waveform, sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    log_mel = waveform_to_logmel(waveform, sample_rate=sr)
    log_mel = log_mel.unsqueeze(0)  # [1, 1, n_mels, time_frames]
    log_mel_np = log_mel.cpu().numpy()

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    outputs = session.run([output_name], {input_name: log_mel_np})
    return outputs[0]  # [1, D]


def cosine_similarity_numpys(emb1: np.ndarray, emb2: np.ndarray) -> float:
    v1 = emb1.flatten()
    v2 = emb2.flatten()
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)


# === MAIN ===
session = load_and_verify_model(ONNX_MODEL_PATH)

embed1 = run_inference_onnx(session, "test00.wav")
embed2 = run_inference_onnx(session, "test01.wav")
embed3 = run_inference_onnx(session, "test02.wav")

print(f"Similarity (1 vs 2): {cosine_similarity_numpys(embed1, embed2)}")
print(f"Similarity (1 vs 3): {cosine_similarity_numpys(embed1, embed3)}")

print("done")
