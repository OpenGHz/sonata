# Temporal Fusion Model I/O And Internal Flow

## 1. Required input fields

The temporal version of PTv3 still consumes the normal point-cloud fields and additionally understands two temporal metadata tensors.

### Base fields

- `coord`: shape `(N, 3)`
- `feat`: shape `(N, C_in)`
- `offset`: shape `(B,)`

Or, if the data is built through the default transform pipeline, the model can also use:

- `grid_coord`: shape `(N, 3)`
- `batch`: shape `(N,)`

### Temporal fields

- `sequence`: shape `(N,)`
  - integer sequence id for each point
  - points from different video clips or temporal streams must have different ids
- `frame`: shape `(N,)`
  - integer frame index for each point
  - larger value means newer frame within the same sequence

## 2. Multi-sequence batch layout

The model works on a flattened point set over all sequences and all frames.

If there are `S` sequences, `T` frames per sequence, and each transformed frame has `N_f` points, then the flattened input typically looks like:

- total points: `N = sum over all frames of N_f = S * T * N_f`
- `coord`: `(N, 3)`
- `feat`: `(N, C_in)`
- `sequence`: `(N,)`
- `frame`: `(N,)`
- `offset`: `(S * T,)` when each frame is collated as one batch item

Example for `S = 2`, `T = 3`, `N_f = 216` in the offline test:

- total flattened points: `1296`
- latest-frame output points after temporal selection: `2 * 216 = 432`

## 3. Internal processing flow

### Step 1. Embedding

`PointTransformerV3.forward()` first wraps the dict into a `Point` object and applies the linear embedding layer:

- input `feat`: `(N, C_in)`
- embedded `feat`: `(N, C_stage0)`

### Step 2. Serialization and sparsification

The point cloud is serialized into PTv3 patch order and converted to sparse spconv tensors.

- serialization uses `grid_coord` and `batch`
- this stage is still spatial only

### Step 3. Spatial attention

Inside each enabled PTv3 block, serialized spatial attention is computed exactly as before:

- points are reordered into serialized patches
- each patch is processed with self-attention over spatial tokens

### Step 4. Temporal attention

When `enable_temporal=True` and the block index matches `temporal_every`, the temporal branch runs in the same block.

The temporal branch does the following:

1. Read `sequence` and `frame` from the current `Point`.
2. Compute a fixed sinusoidal temporal embedding from relative frame index.
3. Reuse the same `qkv` projection already used by spatial attention.
4. Group tokens by:
   - same `sequence`
   - same `grid_coord`
5. For each group, apply causal attention over time only.
6. Add the temporal result back to the spatial-attention output.

This means the effective fusion pattern is:

```text
output = spatial_attention(point) + temporal_attention(point)
```

## 4. Hierarchical metadata propagation

During pooling, `sequence` and `frame` are copied to the pooled parent point set using the head index of each voxel cluster.

This is necessary so temporal grouping remains available in deeper encoder stages.

## 5. Latest-frame output selection

If `temporal_return_current=True`, the model runs `get_current_frame()` after the encoder or decoder output.

For each unique `sequence` id:

1. find the maximum `frame`
2. keep only points from that latest frame
3. rebuild `batch` for the kept sequences
4. reset kept `frame` values to zero in the returned `Point`

As a result, the returned tensor shapes become:

- `coord`: `(N_latest, 3)`
- `feat`: `(N_latest, C_out)`
- `sequence`: `(N_latest,)`
- `frame`: `(N_latest,)`, all zeros after selection

where `N_latest` is the total point count of the latest frame from every sequence.

## 6. Demo-specific flow in `demo/3_batch_forward.py`

The multi-sequence demo performs these steps:

1. Build `num_sequences * num_frames` frame items.
2. Assign per-point `sequence` and `frame`.
3. Collate all frames into one flattened batch.
4. Run PTv3 with temporal fusion.
5. Restore point scale when pooling traces are available.
6. Select the latest frame from each sequence.
7. Visualize:
   - all input frames
   - fused features from the latest frame of every sequence

## 7. Practical constraints

- Real pre-trained temporal runs still depend on checkpoint download and normally expect CUDA for the full PTv3 + spconv path.
- The offline smoke test uses synthetic data and random initialization so the temporal data path can be validated without network access.