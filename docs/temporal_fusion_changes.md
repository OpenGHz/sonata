# Temporal Fusion Changes

## 1. Model changes

- Added a MEM-style temporal branch to PTv3 in `sonata/model.py`.
- The new branch keeps the original serialized spatial attention unchanged and additively fuses a causal temporal attention result.
- Temporal attention reuses the existing `qkv` projection and attention weights, so it does not introduce new learnable parameters.
- Added fixed sinusoidal temporal position encoding with zero value at the latest frame so single-frame behavior stays unchanged.
- Added `enable_temporal`, `temporal_every`, and `temporal_return_current` to `PointTransformerV3`.
- Propagated `sequence` and `frame` metadata through `GridPooling` so temporal fusion still works after hierarchy changes.
- Added `PointTransformerV3.get_current_frame()` to optionally drop old frames after fusion and keep only the latest frame per sequence.

## 2. Demo changes

- Added `demo/4_temporal_fusion.py` as a minimal offline verification script for the temporal branch.
- Reworked `demo/3_batch_forward.py` from a static two-sample batch demo into a temporal forward demo.
- `demo/3_batch_forward.py` now supports:
  - multi-frame input
  - multi-sequence batch input
  - real pre-trained model path
  - offline synthetic smoke-test path
  - optional visualization disable flag for headless environments
  - explicit CPU / CUDA selection
- Added latest-frame filtering and PCA visualization for the fused output.
- Visualization now separates samples by both frame index and sequence index.

## 3. Documentation and environment

- Updated `README.md` with temporal-fusion input fields, config flags, and demo commands.
- Added `sequence` and `frame` field descriptions to `sonata/structure.py`.
- Added missing pixi dependencies needed for the demo workflow earlier in this task, including `addict` and `scipy`.

## 4. Validation

Validated offline with:

```bash
pixi run -e demo python demo/3_batch_forward.py --synthetic --random-init --no-vis --device cpu --num-sequences 2 --num-frames 3
```

Observed output:

```text
Device: cpu
Input sequences: 2
Input frames: 3
Latest-frame points: 432
Latest feature shape: (432, 32)
```