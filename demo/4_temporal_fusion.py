# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sonata
import torch


def build_synthetic_frame(frame_id, grid_size=6):
    axis = torch.linspace(0.0, 1.0, steps=grid_size)
    coord = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    coord = coord.reshape(-1, 3)
    grid_coord = torch.round(coord * (grid_size - 1)).long()
    feat = torch.cat(
        [coord, torch.sin(coord * (frame_id + 1.0) * 3.1415926)], dim=1
    ).float()
    num_points = coord.shape[0]
    return {
        "coord": coord,
        "grid_coord": grid_coord,
        "feat": feat,
        "offset": torch.tensor([num_points], dtype=torch.long),
        "sequence": torch.zeros(num_points, dtype=torch.long),
        "frame": torch.full((num_points,), frame_id, dtype=torch.long),
    }


def build_temporal_batch(num_frames=3):
    frames = []
    for frame_id in range(num_frames):
        frame = build_synthetic_frame(frame_id)
        frame["feat"][:, 3:] = frame["feat"][:, 3:] * (1.0 - 0.1 * frame_id)
        frames.append(frame)
    return sonata.data.collate_fn(frames), frames[-1]


def build_tiny_temporal_ptv3(in_channels):
    return sonata.model.PointTransformerV3(
        in_channels=in_channels,
        stride=(2,),
        enc_depths=(1, 1),
        enc_channels=(32, 64),
        enc_num_head=(4, 8),
        enc_patch_size=(128, 128),
        dec_depths=(1,),
        dec_channels=(32,),
        dec_num_head=(4,),
        dec_patch_size=(128,),
        enable_flash=False,
        enable_temporal=True,
        temporal_every=1,
        temporal_return_current=True,
    )


def prepare_model_for_device(model, batch, device):
    if device.type == "cuda":
        model = model.to(device)
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
        return model, batch

    for module in model.modules():
        if isinstance(module, sonata.model.Block):
            module.cpe = sonata.module.PointSequential(torch.nn.Identity())
    return model, batch


if __name__ == "__main__":
    sonata.utils.set_seed(20260309)

    batch, latest_frame = build_temporal_batch(num_frames=3)
    model = build_tiny_temporal_ptv3(in_channels=batch["feat"].shape[1])
    device = torch.device("cpu")
    model, batch = prepare_model_for_device(model, batch, device)
    model.eval()

    with torch.inference_mode():
        output = model(batch)

    expected_points = latest_frame["coord"].shape[0]
    actual_points = output["coord"].shape[0]

    assert actual_points == expected_points, (
        f"Expected {expected_points} points for the latest frame, got {actual_points}."
    )
    assert "sequence" in output and output["sequence"].unique().numel() == 1
    assert "frame" in output and torch.all(output["frame"] == 0)

    print("Temporal fusion example passed.")
    print(f"Device: {device}")
    print(f"Input frames: 3, latest-frame points: {expected_points}")
    print(f"Output feature shape: {tuple(output['feat'].shape)}")