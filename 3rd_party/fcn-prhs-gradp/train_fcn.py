#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
import time
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import yaml

from fcn_prhs_gradp import ElementwiseFCN, load_element_dataset


# Legacy CLI-only training used direct arguments such as --epochs, --lr, and
# --hidden. New experiments use config.yaml plus --set overrides so FCN studies
# follow the Dist-GNN-style configuration layout.


def parse_value(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if "," in value:
        return [parse_value(item) for item in value.split(",")]
    return value


def apply_override(cfg: dict[str, Any], expr: str) -> None:
    if "=" not in expr:
        raise ValueError(f"Override must be KEY=VALUE: {expr}")
    key, value = expr.split("=", 1)
    cursor = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = parse_value(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an element-wise FCN from prhs to gradp.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override config values, e.g. model.hidden_channels=128")
    return parser.parse_args()


def load_config(path: Path, overrides: list[str]) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    cfg = deepcopy(cfg)
    for override in overrides:
        apply_override(cfg, override)
    return cfg


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def phase_lr(cfg: dict[str, Any], step: int) -> float:
    schedule = cfg["learning_rate_schedule"]
    phase1 = int(schedule["phase1_steps"])
    phase2 = int(schedule["phase2_steps"])
    if step <= phase1:
        return float(schedule["lr_phase12"])
    if step <= phase1 + phase2:
        return float(schedule["lr_phase23"])
    return float(schedule["lr_phase23"])


def total_steps(cfg: dict[str, Any]) -> int:
    schedule = cfg["learning_rate_schedule"]
    return int(schedule["phase1_steps"]) + int(schedule["phase2_steps"]) + int(schedule["phase3_steps"])


def next_batch(loader: DataLoader, iterator: Any) -> tuple[torch.Tensor, torch.Tensor, Any]:
    try:
        x, y = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        x, y = next(iterator)
    return x, y, iterator


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    total_count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        loss = loss_fn(model(x), y)
        total_loss += loss.item() * x.shape[0]
        total_count += x.shape[0]
    return total_loss / max(total_count, 1)


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_history(csv_path: Path, png_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plot because matplotlib is unavailable: {exc}")
        return

    rows: list[dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows.append({key: float(value) for key, value in row.items()})
    if not rows:
        return
    steps = [row["step"] for row in rows]
    train = [row["train_mse"] for row in rows]
    val = [row["val_mse"] for row in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, train, label="train_mse")
    ax.plot(steps, val, label="val_mse")
    ax.set_xlabel("step")
    ax.set_ylabel("normalized MSE")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    general = cfg["general"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    optim_cfg = cfg["optimizer"]
    output_cfg = cfg["output"]

    torch.manual_seed(int(general["seed"]))
    device = select_device(str(general["device"]))

    output_dir = Path(output_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)

    dataset, stats, pairs = load_element_dataset(
        data_dir=Path(data_cfg["gnn_outputs_path"]),
        input_field=str(data_cfg["input_fld_name"]),
        output_field=str(data_cfg["output_fld_name"]),
        input_dim=int(data_cfg["input_fld_dim"]),
        output_dim=int(data_cfg["output_fld_dim"]),
        time_min=data_cfg.get("time_min"),
        time_max=data_cfg.get("time_max"),
        max_samples=data_cfg.get("max_samples"),
        seed=int(general["seed"]),
    )
    n_val = int(len(dataset) * float(data_cfg["val_fraction"]))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(int(general["seed"])),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=int(optim_cfg["batch_size"]),
        shuffle=bool(optim_cfg["shuffle"]),
        num_workers=int(optim_cfg.get("num_workers", 0)),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(optim_cfg["val_batch_size"]),
        shuffle=False,
        num_workers=int(optim_cfg.get("num_workers", 0)),
    )

    hidden = tuple([int(model_cfg["hidden_channels"])] * int(model_cfg["n_mlp_hidden_layers"]))
    model = ElementwiseFCN(
        input_dim=int(data_cfg["input_fld_dim"]),
        output_dim=int(data_cfg["output_fld_dim"]),
        hidden=hidden,
        activation=str(model_cfg["activation"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate_schedule"]["lr_phase12"]),
        weight_decay=float(optim_cfg["weight_decay"]),
    )
    loss_fn = nn.MSELoss()

    print(f"data_dir={data_cfg['gnn_outputs_path']}")
    print(f"pairs={len(pairs)} samples={len(dataset)} train={n_train} val={n_val}")
    print(f"time_range={pairs[0].time}..{pairs[-1].time} device={device}")
    print(f"hidden_channels={model_cfg['hidden_channels']} n_mlp_hidden_layers={model_cfg['n_mlp_hidden_layers']}")
    print(f"x_mean={stats.x_mean.numpy().reshape(-1)} x_std={stats.x_std.numpy().reshape(-1)}")
    print(f"y_mean={stats.y_mean.numpy().reshape(-1)} y_std={stats.y_std.numpy().reshape(-1)}")

    best_val = float("inf")
    history: list[dict[str, Any]] = []
    iterator = iter(train_loader)
    max_steps = total_steps(cfg)
    logfreq = int(general["logfreq"])
    ckptfreq = int(general["ckptfreq"])
    start = time.time()

    for step in range(1, max_steps + 1):
        lr = phase_lr(cfg, step)
        set_lr(optimizer, lr)
        x, y, iterator = next_batch(train_loader, iterator)
        x = x.to(device)
        y = y.to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

        if step % logfreq == 0 or step == 1 or step == max_steps:
            val_loss = eval_epoch(model, val_loader, device) if n_val else float("nan")
            elapsed = time.time() - start
            row = {
                "step": step,
                "train_mse": float(loss.item()),
                "val_mse": float(val_loss),
                "lr": lr,
                "elapsed_sec": elapsed,
            }
            history.append(row)
            print(
                f"[STEP {step}] train_mse={row['train_mse']:.8e} "
                f"val_mse={row['val_mse']:.8e} lr={lr:.3e} elapsed={elapsed:.3f}s"
            )
            write_history(output_dir / "loss_history.csv", history)
            if n_val and val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "stats": stats.as_dict(),
                        "config": cfg,
                        "best_val_mse": best_val,
                    },
                    output_dir / "best.pt",
                )

        if ckptfreq > 0 and step % ckptfreq == 0:
            torch.save(
                {"model_state_dict": model.state_dict(), "stats": stats.as_dict(), "config": cfg, "step": step},
                output_dir / f"step_{step}.pt",
            )

    torch.save(
        {"model_state_dict": model.state_dict(), "stats": stats.as_dict(), "config": cfg, "step": max_steps},
        output_dir / "last.pt",
    )
    plot_history(output_dir / "loss_history.csv", output_dir / "loss_history.png")


if __name__ == "__main__":
    main()
