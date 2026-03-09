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


import argparse
import copy
import os

import open3d as o3d
import sonata
import torch

try:
    import flash_attn
except ImportError:
    flash_attn = None


def get_pca_color(feat, brightness=1.25, center=True):
    u, s, v = torch.pca_lowrank(feat, center=center, q=6, niter=5)
    projection = feat @ v
    projection = projection[:, :3] * 0.6 + projection[:, 3:6] * 0.4
    min_val = projection.min(dim=-2, keepdim=True)[0]
    max_val = projection.max(dim=-2, keepdim=True)[0]
    div = torch.clamp(max_val - min_val, min=1e-6)
    color = (projection - min_val) / div * brightness
    color = color.clamp(0.0, 1.0)
    return color


def build_synthetic_frame(frame_id, sequence_id, grid_size=6):
    axis = torch.linspace(0.0, 1.0, steps=grid_size)
    coord = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    coord = coord.reshape(-1, 3)
    coord[:, 1] += sequence_id * 1.5
    grid_coord = torch.round(coord * (grid_size - 1)).long()
    feat = torch.cat(
        [coord, torch.sin(coord * (frame_id + 1.0 + sequence_id) * 3.1415926)], dim=1
    ).float()
    num_points = coord.shape[0]
    return {
        "coord": coord,
        "grid_coord": grid_coord,
        "feat": feat,
        "offset": torch.tensor([num_points], dtype=torch.long),
        "sequence": torch.full((num_points,), sequence_id, dtype=torch.long),
        "frame": torch.full((num_points,), frame_id, dtype=torch.long),
    }


def build_temporal_batch(num_frames, num_sequences, use_synthetic):
    if use_synthetic:
        frames = []
        for sequence_id in range(num_sequences):
            for frame_id in range(num_frames):
                frame = build_synthetic_frame(frame_id, sequence_id)
                frame["feat"][:, 3:] = frame["feat"][:, 3:] * (
                    1.0 - 0.05 * frame_id - 0.03 * sequence_id
                )
                frames.append(frame)
        return sonata.data.collate_fn(frames)

    transform = sonata.transform.default()
    point = sonata.data.load("sample1")
    point.pop("segment200")
    point["segment"] = point.pop("segment20")

    frames = []
    for sequence_id in range(num_sequences):
        for frame_id in range(num_frames):
            frame = copy.deepcopy(point)
            frame["coord"][:, 1] += sequence_id * 1.5
            frame["color"] = (
                frame["color"] * (1.0 - 0.05 * frame_id) + 0.03 * sequence_id
            ).clip(0.0, 1.0)
            frame = transform(frame)
            num_points = frame["coord"].shape[0]
            frame["sequence"] = torch.full((num_points,), sequence_id, dtype=torch.long)
            frame["frame"] = torch.full((num_points,), frame_id, dtype=torch.long)
            frames.append(frame)
    return sonata.data.collate_fn(frames)


def build_model(in_channels, use_random_init):
    if use_random_init:
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
        )

    custom_config = dict(enable_temporal=True, temporal_every=4)
    if flash_attn is None:
        custom_config.update(
            enc_patch_size=[1024 for _ in range(5)],
            enable_flash=False,
        )
    return sonata.load("sonata", repo_id="facebook/sonata", custom_config=custom_config)


def prepare_model_for_device(model, point, device):
    if device.type == "cuda":
        model = model.to(device)
        for key, value in point.items():
            if isinstance(value, torch.Tensor):
                point[key] = value.to(device)
        return model, point

    for module in model.modules():
        if isinstance(module, sonata.model.Block):
            module.cpe = sonata.module.PointSequential(torch.nn.Identity())
    return model, point


def restore_original_scale(point):
    if "pooling_parent" not in point.keys():
        return point
    for _ in range(2):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    while "pooling_parent" in point.keys():
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = point.feat[inverse]
        point = parent
    return point


def get_latest_mask(point):
    sequence = point.sequence.long()
    frame = point.frame.long()
    _, sequence_inverse = torch.unique(sequence, sorted=True, return_inverse=True)
    num_sequence = int(sequence_inverse.max().item()) + 1
    sequence_max = torch.empty(num_sequence, device=frame.device, dtype=frame.dtype)
    for index in range(num_sequence):
        sequence_max[index] = frame[sequence_inverse == index].max()
    return frame == sequence_max[sequence_inverse]


def build_visualization_clouds(point, latest_mask, num_frames, num_sequences):
    frame = point.frame.float()
    sequence = point.sequence.float()
    input_coord = point.coord.clone()
    input_coord[:, 0] += frame * 8.0
    input_coord[:, 1] += sequence * 8.0
    color_r = 0.25 + 0.6 * (frame / max(num_frames - 1, 1))
    color_g = 0.25 + 0.6 * (sequence / max(num_sequences - 1, 1))
    color_b = torch.full_like(color_r, 0.45)
    frame_color = torch.stack([color_r, color_g, color_b], dim=-1)
    input_pcd = o3d.geometry.PointCloud()
    input_pcd.points = o3d.utility.Vector3dVector(input_coord.cpu().numpy())
    input_pcd.colors = o3d.utility.Vector3dVector(frame_color.cpu().numpy())

    latest_coord = point.coord[latest_mask].clone()
    latest_coord[:, 0] += (num_frames + 1) * 8.0
    latest_coord[:, 1] += point.sequence[latest_mask].float() * 8.0
    latest_color = get_pca_color(point.feat[latest_mask], brightness=1.2, center=True)
    latest_pcd = o3d.geometry.PointCloud()
    latest_pcd.points = o3d.utility.Vector3dVector(latest_coord.cpu().numpy())
    latest_pcd.colors = o3d.utility.Vector3dVector(latest_color.cpu().numpy())
    return input_pcd, latest_pcd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-frames", type=int, default=3)
    parser.add_argument("--num-sequences", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-vis", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--random-init", action="store_true")
    args = parser.parse_args()

    sonata.utils.set_seed(53124)
    point = build_temporal_batch(
        args.num_frames,
        args.num_sequences,
        use_synthetic=args.synthetic,
    )
    model = build_model(point["feat"].shape[1], use_random_init=args.random_init)

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, point = prepare_model_for_device(model, point, device)
    model.eval()

    with torch.inference_mode():
        point = model(point)
        point = restore_original_scale(point)
        latest_mask = get_latest_mask(point)
        latest_feat = point.feat[latest_mask]

    print(f"Device: {device}")
    print(f"Input sequences: {args.num_sequences}")
    print(f"Input frames: {args.num_frames}")
    print(f"Latest-frame points: {int(latest_mask.sum().item())}")
    print(f"Latest feature shape: {tuple(latest_feat.shape)}")

    if args.no_vis or not os.environ.get("DISPLAY"):
        raise SystemExit(0)

    input_pcd, latest_pcd = build_visualization_clouds(
        point,
        latest_mask,
        args.num_frames,
        args.num_sequences,
    )
    o3d.visualization.draw_geometries([input_pcd, latest_pcd])
