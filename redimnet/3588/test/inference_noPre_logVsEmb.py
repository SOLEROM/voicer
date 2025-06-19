#!/usr/bin/env python3
"""
compare_logmel_to_ref_fp16.py
-----------------------------
Compare a pre-computed log-Mel tensor (.npy) against a reference embedding
(.pt) using a ReDimNet-NoMel RKNN – **all data passed as float16**.

Usage
-----
python compare_logmel_to_ref_fp16.py \
       model.rknn  logmel_X  embed_X  [rk3588]

Notes
-----
• `logmel.npy` must be float32/float16 **NCHW** with shape [B, 1, 60, 134].
• `ref_embed.pt` must be saved with `torch.save(tensor, path)` and is
  converted to float16 on load.
• Cosine similarity is computed in float32 for accuracy.
"""

# ───────────────────────── Imports ─────────────────────────
import os
import sys
import numpy as np
from rknn.api import RKNN
import torch                     # only for torch.load()

# os.environ["RKNN_LOG_LEVEL"] = "3"
_EPS = 1e-6

# ─────────────────── helper: log-Mel loader ───────────────────
def load_logmel(path: str) -> np.ndarray:
    """
    Load .npy tensor and return float16 **NCHW** [B, 1, 60, 134].
    Assumes the stored tensor is already NCHW.
    """
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 4 or arr.shape[1] != 1:
        raise ValueError(f"{path}: expected NCHW [B,1,60,134], got {arr.shape}")
    return arr.astype(np.float16, copy=False)

# ─────────────────── helper: reference loader ───────────────────
def load_ref_embedding(pt_path: str) -> np.ndarray:
    ref = torch.load(pt_path, map_location="cpu")
    if isinstance(ref, torch.Tensor):
        ref = ref.cpu().to(torch.float16).numpy()
    else:
        ref = np.asarray(ref, dtype=np.float16)
    return ref

# ─────────────────── cosine similarity ───────────────────
def cosine_similarity(a16: np.ndarray, b16: np.ndarray) -> float:
    a = a16.astype(np.float32, copy=False).flatten()
    b = b16.astype(np.float32, copy=False).flatten()
    return float(np.dot(a, b) /
                 (np.linalg.norm(a) * np.linalg.norm(b) + _EPS))

# ──────────────────── main routine ────────────────────
def main(rknn_model: str, npy_path: str, ref_pt: str, target: str):
    print(f"[1/4] Loading RKNN model: {rknn_model}")
    rk = RKNN()
    if rk.load_rknn(rknn_model) != 0:
        raise RuntimeError("❌ Failed to load RKNN model")

    print(f"[2/4] Initialising runtime for target: {target}")
    if rk.init_runtime(target=target) != 0:
        raise RuntimeError("❌ Failed to initialise RKNN runtime")

    logmel16 = load_logmel(npy_path)      # float16 NCHW
    print(f"[INFO] log-Mel tensor shape {logmel16.shape}, dtype {logmel16.dtype}")

    print("[3/4] Running inference …")
    probe_emb16 = rk.inference(inputs=[logmel16], data_format='nchw')[0] \
                    .astype(np.float16, copy=False)
    rk.release()

    ref_emb16 = load_ref_embedding(ref_pt)
    print(f"[INFO] Reference embedding shape {ref_emb16.shape}, dtype {ref_emb16.dtype}")

    sim = cosine_similarity(probe_emb16, ref_emb16)
    print(f"\n✅ Cosine Similarity : {sim:.4f}")
    print(f"🔎 Distance (1 - sim): {1 - sim:.4f}")

# ──────────────────── CLI wrapper ────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    model_file  = sys.argv[1]
    logmel_npy  = sys.argv[2]
    ref_pt      = sys.argv[3]
    target_chip = sys.argv[4] if len(sys.argv) > 4 else "rk3588"

    main(model_file, logmel_npy, ref_pt, target_chip)
