from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import TensorDataset


FIELD_RE = re.compile(
    r"^fld_(?P<field>[^_]+)_time_(?P<time>[^_]+)_rank_(?P<rank>\d+)_size_(?P<size>\d+)\.bin$"
)


@dataclass(frozen=True)
class SequencePairIndex:
    time: int
    rank: int
    size: int
    x_path: Path
    y_path: Path


@dataclass(frozen=True)
class DatasetStats:
    x_mean: torch.Tensor
    x_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "x_mean": self.x_mean,
            "x_std": self.x_std,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
        }


def _parse_field_file(path: Path, field: str) -> tuple[int, int, int] | None:
    match = FIELD_RE.match(path.name)
    if not match or match.group("field") != field:
        return None
    return (
        int(float(match.group("time"))),
        int(match.group("rank")),
        int(match.group("size")),
    )


def discover_pairs(
    data_dir: Path,
    input_field: str = "prhs",
    output_field: str = "gradp",
    time_min: int | None = None,
    time_max: int | None = None,
    ranks: Iterable[int] | None = None,
) -> list[SequencePairIndex]:
    rank_filter = set(ranks) if ranks is not None else None
    inputs: dict[tuple[int, int, int], Path] = {}
    outputs: dict[tuple[int, int, int], Path] = {}
    for path in Path(data_dir).glob("fld_*_time_*_rank_*_size_*.bin"):
        key_x = _parse_field_file(path, input_field)
        key_y = _parse_field_file(path, output_field)
        key = key_x if key_x is not None else key_y
        if key is None:
            continue
        time, rank, _ = key
        if rank_filter is not None and rank not in rank_filter:
            continue
        if time_min is not None and time < time_min:
            continue
        if time_max is not None and time > time_max:
            continue
        if key_x is not None:
            inputs[key_x] = path
        if key_y is not None:
            outputs[key_y] = path

    missing_outputs = sorted(set(inputs) - set(outputs))
    missing_inputs = sorted(set(outputs) - set(inputs))
    if missing_outputs or missing_inputs:
        raise RuntimeError(
            "Unpaired field files: "
            f"missing_outputs={missing_outputs[:5]} missing_inputs={missing_inputs[:5]}"
        )

    return [
        SequencePairIndex(time=key[0], rank=key[1], size=key[2], x_path=inputs[key], y_path=outputs[key])
        for key in sorted(inputs)
    ]


def _read_pair(pair: SequencePairIndex, input_dim: int, output_dim: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.fromfile(pair.x_path, dtype=np.float64).reshape((-1, input_dim)).astype(np.float32)
    y = np.fromfile(pair.y_path, dtype=np.float64).reshape((-1, output_dim)).astype(np.float32)
    n = min(x.shape[0], y.shape[0])
    return x[:n], y[:n]


def load_sequence_dataset(
    data_dir: Path,
    input_field: str = "prhs",
    output_field: str = "gradp",
    input_dim: int = 1,
    output_dim: int = 3,
    sequence_length: int = 5,
    stride: int = 1,
    time_min: int | None = 5000,
    time_max: int | None = 25000,
    ranks: Iterable[int] | None = None,
    max_samples: int | None = None,
    seed: int = 12,
) -> tuple[TensorDataset, DatasetStats, list[SequencePairIndex]]:
    pairs = discover_pairs(
        data_dir=Path(data_dir),
        input_field=input_field,
        output_field=output_field,
        time_min=time_min,
        time_max=time_max,
        ranks=ranks,
    )
    if not pairs:
        raise RuntimeError(f"No paired files found in {data_dir}")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")

    grouped: dict[tuple[int, int], list[SequencePairIndex]] = {}
    for pair in pairs:
        grouped.setdefault((pair.rank, pair.size), []).append(pair)

    seqs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for _, group in sorted(grouped.items()):
        group = sorted(group, key=lambda item: item.time)
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for pair in group:
            x, y = _read_pair(pair, input_dim=input_dim, output_dim=output_dim)
            xs.append(x)
            ys.append(y)
        x_time = np.stack(xs, axis=0)
        y_time = np.stack(ys, axis=0)
        for start in range(0, len(group) - sequence_length + 1, stride):
            end = start + sequence_length
            seqs.append(np.transpose(x_time[start:end], (1, 0, 2)))
            targets.append(y_time[end - 1])

    x_all = np.concatenate(seqs, axis=0)
    y_all = np.concatenate(targets, axis=0)
    if max_samples is not None and max_samples < x_all.shape[0]:
        rng = np.random.default_rng(seed)
        keep = rng.choice(x_all.shape[0], size=max_samples, replace=False)
        x_all = x_all[keep]
        y_all = y_all[keep]

    x = torch.from_numpy(x_all)
    y = torch.from_numpy(y_all)
    stats = DatasetStats(
        x_mean=x.reshape(-1, input_dim).mean(dim=0, keepdim=True),
        x_std=x.reshape(-1, input_dim).std(dim=0, keepdim=True, unbiased=False),
        y_mean=y.mean(dim=0, keepdim=True),
        y_std=y.std(dim=0, keepdim=True, unbiased=False),
    )
    small = torch.tensor(1.0e-12, dtype=x.dtype)
    x_scaled = (x - stats.x_mean.view(1, 1, -1)) / (stats.x_std.view(1, 1, -1) + small)
    y_scaled = (y - stats.y_mean) / (stats.y_std + small)
    return TensorDataset(x_scaled, y_scaled), stats, pairs

