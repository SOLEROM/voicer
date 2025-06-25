#!/usr/bin/env python3
"""
convert_onnx_to_rknn.py – ONNX → RKNN (+ overflow / NaN simulator check)

Quick start
-----------
python convert_onnx_to_rknn.py  model.onnx  rk3588

Full usage
----------
python convert_onnx_to_rknn.py \
       model.onnx  <platform>  [dtype]  [out.rknn]  [logmel.npy]  [embed.*]

Arguments
---------
platform   – rk3562 | rk3566 | rk3568 | rk3576 | rk3588 |
             rv1103 | rv1106 | rk1808 | rv1109 | rv1126

dtype      – fp | i8 | u8       (default fp → FP16 on NPU)

out.rknn   – output file (default model.rknn)

logmel.npy – optional input log-Mel tensor:
             • NCHW  [1, 1, 60, 134]  or
             • NHWC  [1, 60, 134, 1]

embed.*    – optional reference embedding:
             • .npy   (np.save)
             • .npz   (np.savez / np.savez_compressed)   – key 'embed' or single array
             • .pt / .pth / .torch  (torch.save)

If **both** `logmel.npy` **and** `embed.*` are supplied, the script:
  1. Runs simulator `accuracy_analysis` on the sample.
  2. Runs host-CPU inference.
  3. Compares RKNN output to the reference (Δmax/Δmean & cosine sim).
"""

from __future__ import annotations
import sys, os, atexit, tempfile
import numpy as np
from rknn.api import RKNN

# ───────────────── defaults ─────────────────
DATASET_TXT   = './cal/dataset.txt'
DEFAULT_OUT   = 'model.rknn'
DEFAULT_DTYPE = 'fp'
FRAMES        = 134
VALID_PLATFORMS = {
    'rk3562','rk3566','rk3568','rk3576','rk3588',
    'rv1103','rv1106','rk1808','rv1109','rv1126'
}
# ────────────────────────────────────────────


# ───────────── CLI parsing ──────────────
def parse_cli():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)

    onnx_path = sys.argv[1]
    platform  = sys.argv[2].lower()
    if platform not in VALID_PLATFORMS:
        sys.exit(f'ERROR: unknown platform {platform}')

    dtype   = sys.argv[3].lower() if len(sys.argv) > 3 else DEFAULT_DTYPE
    if dtype not in ('fp', 'i8', 'u8'):
        sys.exit('ERROR: dtype must be fp / i8 / u8')

    out_rknn   = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_OUT
    logmel_npy = sys.argv[5] if len(sys.argv) > 5 else None
    embed_path = sys.argv[6] if len(sys.argv) > 6 else None

    if (logmel_npy is None) ^ (embed_path is None):
        sys.exit('ERROR: supply both logmel.npy and embed file, or neither.')

    do_quant = dtype in ('i8', 'u8')
    return onnx_path, platform, do_quant, out_rknn, logmel_npy, embed_path


# ───── helper: tmp random sample ─────
def random_sample_path() -> str:
    arr = np.random.randn(1, 1, 60, FRAMES).astype(np.float32)
    f   = tempfile.NamedTemporaryFile(suffix='.npy', delete=False,
                                      prefix='_tmp_sample_', dir='.')
    np.save(f.name, arr); f.close()
    atexit.register(lambda: os.remove(f.name))
    return f.name


# ───── helper: load / fix log-Mel ─────
def load_logmel(path: str) -> np.ndarray:
    x = np.load(path)
    if x.dtype != np.float16:
        x = x.astype(np.float16)

    if x.ndim == 3:                       # add batch
        x = x[None, ...]

    if x.shape[1] != 1:                   # assume NHWC → NCHW
        if x.shape[-1] != 1 or x.shape[1] != 60:
            raise ValueError(f'Cannot interpret log-Mel shape {x.shape}')
        x = x.transpose(0, 3, 1, 2)

    if x.shape[2] != 60 or x.shape[3] != FRAMES:
        raise ValueError(f'Expected [1,1,60,{FRAMES}], got {x.shape}')
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
        # still continue: we'll compare up to the common length
        min_len = min(rknn_flat.size, ref_flat.size)
        rknn_flat = rknn_flat[:min_len]
        ref_flat  = ref_flat [:min_len]

    diff = np.abs(rknn_flat - ref_flat)
    cos  = float(np.dot(rknn_flat, ref_flat) /
                 (np.linalg.norm(rknn_flat) * np.linalg.norm(ref_flat) + 1e-9))

    print(f'🔎  Δmax={diff.max():.6f}  Δmean={diff.mean():.6f}  '
          f'cosine={cos:.6f}')
    if np.allclose(rknn_flat, ref_flat, atol=atol):
        print('✅ RKNN output matches reference within tolerance.')
    else:
        print('❌ RKNN output differs from reference.')



# ───────────────────── main ─────────────────────
def main():
    (onnx_path, platform, do_quant, out_rknn,
     logmel_npy, embed_path) = parse_cli()

    rknn = RKNN(verbose=True)

    # 1. config
    # Valid input is ['mean_values', 'std_values', 'quantized_dtype', 'quantized_algorithm', 
    # 'quantized_method', 'quantized_hybrid_level', 'target_platform', 'quant_img_RGB2BGR', 
    # 'float_dtype', 'optimization_level', 'custom_string', 'remove_weight', 'compress_weight', 
    # 'inputs_yuv_fmt', 'single_core_mode', 'dynamic_input', 'model_pruning', 'op_target',
    # 'quantize_weight', 'remove_reshape', 'sparse_infer', 'enable_flash_attention', 
    # 'auto_hybrid_cos_thresh', 'auto_hybrid_euc_thresh', 'output_optimize', 'set_op_attr', 
    # 'sram_prefer', 'nbuf_prefer', 'check_data', 'stream_op', 'disable_rules', 'enable_rnn_loop'].
    print('[1/7] config()')
    rknn.config(mean_values=[[0]], std_values=[[1]],
                target_platform=platform, optimization_level=0)

    # 2. load ONNX
    print('[2/7] load_onnx() →', onnx_path)
    if rknn.load_onnx(onnx_path,
                      inputs=['log_mel'],
                      input_size_list=[[1, 1, 60, FRAMES]]) != 0:
        sys.exit('load_onnx failed')

    # 3. build
    print('[3/7] build()', 'with quant' if do_quant else 'FP16')
    if rknn.build(do_quantization=do_quant,
                  dataset=DATASET_TXT if do_quant else None) != 0:
        sys.exit('build failed')

    # 4. accuracy analysis
    sample_path = logmel_npy if logmel_npy else random_sample_path()
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
        print("===================================================accuracy")
    except Exception as e:
        print(f'⚠ accuracy_analysis skipped ({e})')
        sys.exit('accuracy_analysis failed')

    # 5. export
    print('[5/7] export_rknn() →', out_rknn)
    if rknn.export_rknn(out_rknn) != 0:
        sys.exit('export failed')

    if not do_quant:
        print("===========================================================")
        print("=============== Simulator inference analysis ==============")
        print("===========================================================")
        # 6. host-CPU runtime
        print('[6/7] init_runtime(target=None) – host CPU')
        if rknn.init_runtime(target=None) != 0:
            sys.exit('init_runtime failed (check Toolkit install)')

        # 7. inference + comparison
        print('[7/7] inference() with',
            logmel_npy if logmel_npy else 'random tensor')
        inp = load_logmel(sample_path)
        outs = rknn.inference(inputs=[inp], data_format='nchw')
        print('Host-CPU inference OK → got', len(outs),
            'output(s); first output shape:', outs[0].shape)
        if logmel_npy and embed_path:
            compare_outputs(outs[0].flatten(), embed_path , atol=0.1)
        print("===========================================================")
        print("===========================================================")
        print("==================================================inference")


    ## TODO
    # print('[/] sanity-check input attr')
    # assert rknn.init_runtime(target=None) == 0, 'init_runtime failed (PC runtime)'
    # any way to quary fro params like on cpp::rknn_query


    print('✔ RKNN saved at', out_rknn)
    rknn.release()
    print('EOF!')



if __name__ == '__main__':
    main()
