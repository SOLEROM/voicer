#!/usr/bin/env python3
"""
compare_logmel_to_ref.py
------------------------
Compare a pre-computed log-Mel tensor (.npy) against a reference embedding
(.pt) using a ReDimNet-NoMel RKNN.

Usage
-----
python compare_logmel_to_ref.py  model.rknn  logmel.npy  ref_embed.pt  [rk3588]

• `logmel.npy` must contain a float32 tensor shaped either
  [1, 1, 60, 200] (NCHW) or [1, 60, 200, 1] (NHWC).
• `ref_embed.pt` must be saved via `torch.save(tensor, path)`.
"""

# ─────────────────────────── Imports ────────────────────────────
import os
import sys
import numpy as np
from rknn.api import RKNN
import torch           # only for torch.load()

os.environ["RKNN_LOG_LEVEL"] = "3"

_EPS = 1e-6            # for cosine similarity

# ──────────────────── helpers ────────────────────
def load_logmel(npy_path: str) -> np.ndarray:
    """
    Returns NHWC tensor as float32, shape [B, 60, 200, 1].
    Accepts either NCHW or NHWC in the file.
    """
    arr = np.load(npy_path, allow_pickle=False)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)

    if arr.ndim != 4:
        raise ValueError(f"Expected 4-D tensor in {npy_path}, got {arr.shape}")

    if arr.shape[-1] == 1:                     # already NHWC
        return arr
    if arr.shape[1] == 1:                      # NCHW → NHWC
        return np.transpose(arr, (0, 2, 3, 1))

    raise ValueError("Tensor layout not recognised "
                     "(expect [B,1,60,200] or [B,60,200,1]).")


def load_reference_embedding(pt_path: str) -> np.ndarray:
    """
    Loads a .pt saved via torch.save(tensor, path) and returns a NumPy array.
    """
    try:
        ref = torch.load(pt_path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Failed to load {pt_path}: {e}")

    if isinstance(ref, torch.Tensor):
        return ref.cpu().numpy()
    return np.asarray(ref, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
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

    # ── Load tensor & run inference ──
    logmel_nhwc = load_logmel(npy_path)
    print(f"[INFO] Loaded log-Mel tensor shape: {logmel_nhwc.shape}")

    print("[3/4] Running inference …")
    probe_emb = rk.inference(inputs=[logmel_nhwc])[0]
    rk.release()

    # ── Load reference embedding ──
    ref_emb = load_reference_embedding(ref_pt)
    print(f"[INFO] Loaded reference embedding shape: {ref_emb.shape}")

    # ── Cosine similarity ──
    sim = cosine_similarity(probe_emb, ref_emb)
    print(f"\n✅ Cosine Similarity : {sim:.4f}")
    print(f"🔎 Distance (1 - sim): {1 - sim:.4f}")

# ──────────────────── CLI wrapper ────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    model_file = sys.argv[1]
    logmel_npy = sys.argv[2]
    ref_pt     = sys.argv[3]
    target_chip = sys.argv[4] if len(sys.argv) > 4 else "rk3588"

    main(model_file, logmel_npy, ref_pt, target_chip)
