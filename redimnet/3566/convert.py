#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert.py – ONNX → RKNN (+ overflow / NaN simulator check)

Quick start
-----------
python convert.py  model.onnx  rk3588  B0 output.rknn
python convert.py  model.onnx  rk3588  B2 output.rknn

Full usage
----------
python convert.py \
    model.onnx  <platform>  [dtype]  [B0|B2]  [out.rknn]  [logmel.npy]  [embed.*]

Arguments
---------
platform   – rk3562 | rk3566 | rk3568 | rk3576 | rk3588 |
             rv1103 | rv1106 | rk1808 | rv1109 | rv1126

dtype      – fp | i8 | u8       (default: fp → FP16 on NPU)

BAND       – B0 | B2            (default: B0 unless inferred from `logmel.npy`)
             • B0 → 60 mel bins  → input [1, 1, 60, 134]  (or NHWC [1, 60, 134, 1])
             • B2 → 72 mel bins  → input [1, 1, 72, 134]  (or NHWC [1, 72, 134, 1])

out.rknn   – output file (default: model.rknn)

logmel.npy – optional input log-Mel tensor (NCHW or NHWC; dtype will be cast to FP16)
             If provided and BAND not set, the script infers B0/B2 from the tensor shape.

embed.*    – optional reference embedding:
             • .npy   (np.save)
             • .npz   (np.savez / np.savez_compressed) – key 'embed' or single array
             • .pt / .pth / .torch  (torch.save)

Behavior when both `logmel.npy` AND `embed.*` are supplied:
  1) Runs simulator `accuracy_analysis` on the sample.
  2) Runs host-CPU inference.
  3) Compares RKNN output to the reference (Δmax/Δmean & cosine sim).

Examples
--------
# FP16 (no quant), B0 band explicitly:
python convert.py model.onnx rk3588 fp B0 out.rknn logmel.npy embed.npy

# INT8 quant, B2 band, with dataset.txt in ./cal/:
python convert.py model.onnx rk3588 i8 B2 out_q.rknn

# Let the script infer B0/B2 from input sample:
python convert.py model.onnx rk3588 fp out.rknn logmel.npy embed.pt
"""

from __future__ import annotations
import sys, os, atexit, tempfile
from enum import Enum
import numpy as np
from rknn.api import RKNN

# ───────────────── defaults ─────────────────
DATASET_TXT     = './cal/dataset.txt'
DEFAULT_OUT     = 'model.rknn'
DEFAULT_DTYPE   = 'fp'
DEFAULT_FRAMES  = 134
DEFAULT_BAND    = 'B0'  # will be overridden by inference from logmel.npy if available

VALID_PLATFORMS = {
    'rk3562','rk3566','rk3568','rk3576','rk3588',
    'rv1103','rv1106','rk1808','rv1109','rv1126'
}
# ────────────────────────────────────────────


# ───────────── Band enum ──────────────
class Band(Enum):
    B0 = 60  # 60 mel bins
    B2 = 72  # 72 mel bins

    @staticmethod
    def from_str(s: str) -> "Band":
        sl = s.strip().lower()
        if sl == 'b0': return Band.B0
        if sl == 'b2': return Band.B2
        raise ValueError('BAND must be B0 or B2')


# ───────────── CLI parsing ──────────────
def parse_cli():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)

    onnx_path = sys.argv[1]
    platform  = sys.argv[2].lower()
    if platform not in VALID_PLATFORMS:
        sys.exit(f'ERROR: unknown platform {platform}')

    # Optional positional fields after <platform>:
    # [dtype] [B0|B2] [out.rknn] [logmel.npy] [embed.*]
    # We parse flexibly: detect a BAND token if present.
    dtype      = DEFAULT_DTYPE
    band_token = None
    out_rknn   = DEFAULT_OUT
    logmel_npy = None
    embed_path = None

    tail = sys.argv[3:]

    # dtype present?
    if tail and tail[0].lower() in ('fp', 'i8', 'u8'):
        dtype = tail.pop(0).lower()

    # BAND present?
    if tail and tail[0].strip().lower() in ('b0', 'b2'):
        band_token = tail.pop(0)

    # out.rknn present?
    if tail:
        out_rknn = tail.pop(0)

    # logmel / embed present?
    if tail:
        logmel_npy = tail.pop(0)
    if tail:
        embed_path = tail.pop(0)
    if tail:
        sys.exit('ERROR: too many positional arguments.')

    if dtype not in ('fp', 'i8', 'u8'):
        sys.exit('ERROR: dtype must be fp / i8 / u8')

    if (logmel_npy is None) ^ (embed_path is None):
        sys.exit('ERROR: supply both logmel.npy and embed file, or neither.')

    do_quant = dtype in ('i8', 'u8')
    band = Band.from_str(band_token) if band_token else None
    return onnx_path, platform, dtype, do_quant, band, out_rknn, logmel_npy, embed_path


# ───── helper: infer BAND from a logmel sample ─────
def infer_band_from_logmel(path: str) -> Band:
    x = np.load(path)
    if x.ndim == 3:
        # [C,H,W] or [H,W,C] → add batch
        x = x[None, ...]
    # Try NCHW first
    if x.ndim == 4:
        # NCHW?
        if x.shape[1] == 1:
            h = x.shape[2]
            if h == Band.B0.value: return Band.B0
            if h == Band.B2.value: return Band.B2
        # NHWC?
        if x.shape[-1] == 1:
            h = x.shape[1]
            if h == Band.B0.value: return Band.B0
            if h == Band.B2.value: return Band.B2
    raise ValueError(f"Cannot infer band from logmel shape {x.shape}; "
                     "please pass BAND explicitly (B0 or B2).")


# ───── helper: tmp random sample ─────
def random_sample_path(mels: int, frames: int = DEFAULT_FRAMES) -> str:
    arr = np.random.randn(1, 1, mels, frames).astype(np.float32)
    f   = tempfile.NamedTemporaryFile(suffix='.npy', delete=False,
                                      prefix='_tmp_sample_', dir='.')
    np.save(f.name, arr); f.close()
    atexit.register(lambda: os.remove(f.name))
    return f.name


# ───── helper: load / fix log-Mel ─────
def load_logmel(path: str, mels: int, frames: int = DEFAULT_FRAMES) -> np.ndarray:
    x = np.load(path)
    if x.dtype != np.float16:
        x = x.astype(np.float16)

    if x.ndim == 3:                       # add batch if needed
        x = x[None, ...]

    if x.shape[1] != 1:                   # assume NHWC → NCHW
        if x.shape[-1] != 1 or x.shape[1] != mels:
            raise ValueError(f'Cannot interpret log-Mel shape {x.shape}')
        x = x.transpose(0, 3, 1, 2)

    if x.shape[2] != mels or x.shape[3] != frames:
        raise ValueError(f'Expected [1,1,{mels},{frames}], got {x.shape}')
    return x


# ───── helper: load embedding (npy / npz / torch) ─────
def load_embedding(path: str) -> np.ndarray:
    """
    Load an embedding saved as .npy / .npz / torch pickle.
    Fixes legacy pickles that reference 'numpy._core' or
    'numpy._core.multiarray'.
    """
    ext = os.path.splitext(path)[1].lower()

    # --- .npy straightforward ---
    if ext == '.npy':
        return np.load(path)

    # --- .npz archive ---
    if ext == '.npz':
        arch = np.load(path)
        keys = list(arch.keys())
        if 'embed' in keys:
            return arch['embed']
        if len(keys) == 1:
            return arch[keys[0]]
        raise ValueError(
            f'Archive {path} has multiple arrays {keys}; '
            "store the wanted one under key 'embed'.")

    # --- torch pickle ---
    if ext in ('.pt', '.pth', '.torch'):
        import sys as _sys, torch, numpy as _np
        # shims for legacy pickle module names
        if 'numpy._core' not in _sys.modules:
            _sys.modules['numpy._core'] = _np
        if 'numpy._core.multiarray' not in _sys.modules:
            _sys.modules['numpy._core.multiarray'] = _np.core.multiarray

        obj = torch.load(path, map_location='cpu')

        # tensor → ndarray
        if hasattr(obj, 'cpu'):
            obj = obj.cpu().numpy()
        elif not isinstance(obj, np.ndarray):
            raise TypeError(f'Unexpected object in {path}: {type(obj)}')

        return obj

    raise ValueError(f'Unsupported embedding file extension: {ext}')


# ───── helper: compare embeddings ─────
def compare_outputs(rknn_out: np.ndarray, embed_path: str, atol=1e-3):
    ref_np = load_embedding(embed_path).astype(rknn_out.dtype)

    # --- bring both to flat 1-D ---
    rknn_flat = rknn_out.reshape(-1)
    ref_flat  = ref_np.reshape(-1)

    if rknn_flat.shape != ref_flat.shape:
        print(f'⚠ shape mismatch – RKNN {rknn_flat.shape}, ref {ref_flat.shape}')
        # Compare up to the common length
        min_len = min(rknn_flat.size, ref_flat.size)
        rknn_flat = rknn_flat[:min_len]
        ref_flat  = ref_flat [:min_len]

    diff = np.abs(rknn_flat - ref_flat)
    cos  = float(np.dot(rknn_flat, ref_flat) /
                 (np.linalg.norm(rknn_flat) * np.linalg.norm(ref_flat) + 1e-9))

    print(f'🔎  Δmax={diff.max():.6f}  Δmean={diff.mean():.6f}  cosine={cos:.6f}')
    if np.allclose(rknn_flat, ref_flat, atol=atol):
        print('✅ RKNN output matches reference within tolerance.')
    else:
        print('❌ RKNN output differs from reference.')


# ───────────────────── main ─────────────────────
def main():
    (onnx_path, platform, dtype, do_quant, band_cli,
     out_rknn, logmel_npy, embed_path) = parse_cli()

    # Resolve BAND
    if band_cli is not None:
        band = band_cli
    elif logmel_npy:
        band = infer_band_from_logmel(logmel_npy)
        print(f'[info] BAND inferred from logmel: {band.name} ({band.value} mels)')
    else:
        band = Band.from_str(DEFAULT_BAND)
        print(f'[info] BAND defaulting to {band.name} ({band.value} mels)')

    mels = band.value
    frames = DEFAULT_FRAMES

    rknn = RKNN(verbose=True)

    # 1. config
    print('[1/7] config()')
    rknn.config(mean_values=[[0]], std_values=[[1]],
                target_platform=platform, optimization_level=0)

    # 2. load ONNX
    print('[2/7] load_onnx() →', onnx_path, f'(input [1,1,{mels},{frames}])')
    if rknn.load_onnx(onnx_path,
                      inputs=['log_mel'],
                      input_size_list=[[1, 1, mels, frames]]) != 0:
        sys.exit('load_onnx failed')

    # 3. build
    print('[3/7] build()', 'with quant' if do_quant else 'FP16')
    if rknn.build(do_quantization=do_quant,
                  dataset=DATASET_TXT if do_quant else None) != 0:
        sys.exit('build failed')

    # 4. accuracy analysis
    sample_path = logmel_npy if logmel_npy else random_sample_path(mels, frames)
    print('[4/7] accuracy_analysis() – simulator run on', sample_path)
    try:
        print("===========================================================")
        print("=============== Simulator accuracy analysis ===============")
        print("===========================================================")
        rknn.accuracy_analysis(inputs=[sample_path], target=None)
        print('↑  Check the table: the first "inf"/"nan" row pinpoints the layer '
              'whose activations exceed ±65 k in FP16.\n')
        print("===========================================================")
        print("===========================================================")
        print("===========================================================")
    except Exception as e:
        print(f'⚠ accuracy_analysis skipped ({e})')
        sys.exit('accuracy_analysis failed')

    # 5. export
    print('[5/7] export_rknn() →', out_rknn)
    if rknn.export_rknn(out_rknn) != 0:
        sys.exit('export failed')

    # 6–7. host CPU runtime (only for non-quantized build)
    if dtype == 'fp' and not do_quant:
        print('[6/7] init_runtime(target=None) – host CPU')
        if rknn.init_runtime(target=None) != 0:
            sys.exit('init_runtime failed (check Toolkit install)')

        print('[7/7] inference() with', logmel_npy if logmel_npy else 'random tensor')
        inp = load_logmel(sample_path, mels, frames)
        outs = rknn.inference(inputs=[inp], data_format='nchw')
        print('Host-CPU inference OK → got', len(outs),
              'output(s); first output shape:', outs[0].shape)
        if logmel_npy and embed_path:
            compare_outputs(outs[0].flatten(), embed_path, atol=0.1)

    print('✔ RKNN saved at', out_rknn)
    rknn.release()
    print('EOF!')


if __name__ == '__main__':
    main()
