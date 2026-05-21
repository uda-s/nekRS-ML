#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml


@dataclass
class RunData:
    run_dir: Path
    tag: str
    hidden: int
    n_layers: int
    lr_phase12: float
    lr_phase23: float
    batch_size: int
    phase1_steps: int
    phase2_steps: int
    phase3_steps: int
    steps: list[int]
    train_mse: list[float]
    val_mse: list[float]
    lrs: list[float]

    @property
    def final_step(self) -> int | None:
        return self.steps[-1] if self.steps else None

    @property
    def final_train_mse(self) -> float | None:
        return self.train_mse[-1] if self.train_mse else None

    @property
    def final_val_mse(self) -> float | None:
        return self.val_mse[-1] if self.val_mse else None

    @property
    def best_val_mse(self) -> float | None:
        return min(self.val_mse) if self.val_mse else None


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def read_loss_history(path: Path) -> tuple[list[int], list[float], list[float], list[float]]:
    steps: list[int] = []
    train_mse: list[float] = []
    val_mse: list[float] = []
    lrs: list[float] = []
    with path.open("r", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            steps.append(int(float(row["step"])))
            train_mse.append(float(row["train_mse"]))
            val_mse.append(float(row["val_mse"]))
            lrs.append(float(row["lr"]))
    return steps, train_mse, val_mse, lrs


def parse_run(run_dir: Path) -> RunData | None:
    history_path = run_dir / "loss_history.csv"
    config_path = run_dir / "resolved_config.yaml"
    if not history_path.is_file() or not config_path.is_file():
        return None

    cfg = read_yaml(config_path)
    steps, train_mse, val_mse, lrs = read_loss_history(history_path)
    model = cfg["model"]
    schedule = cfg["learning_rate_schedule"]
    optim = cfg["optimizer"]

    hidden = int(model["hidden_channels"])
    n_layers = int(model["n_mlp_hidden_layers"])
    lr12 = float(schedule["lr_phase12"])
    lr23 = float(schedule["lr_phase23"])
    batch = int(optim["batch_size"])
    ph1 = int(schedule["phase1_steps"])
    ph2 = int(schedule["phase2_steps"])
    ph3 = int(schedule["phase3_steps"])
    tag = f"H{hidden}_L{n_layers}_LR{lr12:g}_{lr23:g}_B{batch}_P{ph1}_{ph2}_{ph3}"

    return RunData(
        run_dir=run_dir,
        tag=tag,
        hidden=hidden,
        n_layers=n_layers,
        lr_phase12=lr12,
        lr_phase23=lr23,
        batch_size=batch,
        phase1_steps=ph1,
        phase2_steps=ph2,
        phase3_steps=ph3,
        steps=steps,
        train_mse=train_mse,
        val_mse=val_mse,
        lrs=lrs,
    )


def discover_runs(root: Path, recursive: bool) -> list[RunData]:
    pattern = "**/loss_history.csv" if recursive else "*/loss_history.csv"
    runs: list[RunData] = []
    for history_path in sorted(root.glob(pattern)):
        run = parse_run(history_path.parent)
        if run is not None:
            runs.append(run)
    return runs


def save_curve(run: RunData, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    if run.steps:
        ax.plot(run.steps, run.train_mse, label="train_mse")
        ax.plot(run.steps, run.val_mse, label="val_mse")
        ax.set_xlabel("step")
        ax.set_ylabel("normalized MSE")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No loss history", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title(run.tag)
    fig.tight_layout()
    fig.savefig(outdir / f"curve.{run.tag}.png", dpi=200)
    plt.close(fig)


def save_curve_csv(run: RunData, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / f"curve.{run.tag}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "train_mse", "val_mse", "lr"])
        for row in zip(run.steps, run.train_mse, run.val_mse, run.lrs):
            writer.writerow(row)


def save_overlay(runs: list[RunData], outpath: Path, metric: str) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for run in sorted(runs, key=lambda r: (r.hidden, r.n_layers, r.lr_phase12, r.lr_phase23)):
        values = run.val_mse if metric == "val_mse" else run.train_mse
        ax.plot(run.steps, values, label=run.tag)
    ax.set_xlabel("step")
    ax.set_ylabel(metric)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def write_summary(runs: list[RunData], outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "run_dir",
                "tag",
                "hidden_channels",
                "n_mlp_hidden_layers",
                "lr_phase12",
                "lr_phase23",
                "batch_size",
                "phase1_steps",
                "phase2_steps",
                "phase3_steps",
                "final_step",
                "final_train_mse",
                "final_val_mse",
                "best_val_mse",
            ]
        )
        for run in sorted(runs, key=lambda r: (r.best_val_mse if r.best_val_mse is not None else float("inf"), r.tag)):
            writer.writerow(
                [
                    str(run.run_dir),
                    run.tag,
                    run.hidden,
                    run.n_layers,
                    run.lr_phase12,
                    run.lr_phase23,
                    run.batch_size,
                    run.phase1_steps,
                    run.phase2_steps,
                    run.phase3_steps,
                    run.final_step,
                    run.final_train_mse,
                    run.final_val_mse,
                    run.best_val_mse,
                ]
            )


def heatmap_2d(
    title: str,
    x_label: str,
    y_label: str,
    x_vals: list[Any],
    y_vals: list[Any],
    z_grid: list[list[float | None]],
    outpath: Path,
) -> None:
    import math

    outpath.parent.mkdir(parents=True, exist_ok=True)
    z_num = [[float("nan") if v is None else v for v in row] for row in z_grid]
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(z_num, origin="lower", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([str(v) for v in x_vals], rotation=45, ha="right")
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels([str(v) for v in y_vals])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("best val_mse")
    for iy, row in enumerate(z_grid):
        for ix, value in enumerate(row):
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            ax.text(ix, iy, f"{value:.2e}", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(outpath, dpi=250)
    plt.close(fig)


def save_heatmaps(runs: list[RunData], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    hiddens = sorted({run.hidden for run in runs})
    layers = sorted({run.n_layers for run in runs})
    lr_pairs = sorted({(run.lr_phase12, run.lr_phase23) for run in runs})

    for lr12, lr23 in lr_pairs:
        selected = [run for run in runs if run.lr_phase12 == lr12 and run.lr_phase23 == lr23]
        index = {(run.hidden, run.n_layers): run for run in selected}
        grid: list[list[float | None]] = []
        for n_layer in layers:
            row: list[float | None] = []
            for hidden in hiddens:
                run = index.get((hidden, n_layer))
                row.append(run.best_val_mse if run else None)
            grid.append(row)
        heatmap_2d(
            title=f"Best val_mse (lr={lr12:g}/{lr23:g})",
            x_label="hidden_channels",
            y_label="n_mlp_hidden_layers",
            x_vals=hiddens,
            y_vals=layers,
            z_grid=grid,
            outpath=outdir / f"heat_hidden_vs_layers__lr_{lr12:g}_{lr23:g}.png",
        )

    for n_layer in layers:
        selected = [run for run in runs if run.n_layers == n_layer]
        index = {(run.hidden, f"{run.lr_phase12:g}/{run.lr_phase23:g}"): run for run in selected}
        lr_labels = [f"{lr12:g}/{lr23:g}" for lr12, lr23 in lr_pairs]
        grid = []
        for lr_label in lr_labels:
            row = []
            for hidden in hiddens:
                run = index.get((hidden, lr_label))
                row.append(run.best_val_mse if run else None)
            grid.append(row)
        heatmap_2d(
            title=f"Best val_mse (layers={n_layer})",
            x_label="hidden_channels",
            y_label="lr_phase12/lr_phase23",
            x_vals=hiddens,
            y_vals=lr_labels,
            z_grid=grid,
            outpath=outdir / f"heat_hidden_vs_lr__layers_{n_layer}.png",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("fcn_outputs_grid"))
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"--root not found: {root}")

    runs = discover_runs(root, recursive=args.recursive)
    if not runs:
        raise RuntimeError(f"No FCN runs found under {root}")

    outroot = args.outdir.resolve() if args.outdir else root / "post"
    curves_dir = outroot / "curves"
    curve_csv_dir = outroot / "curve_csv"
    heat_dir = outroot / "heatmaps"

    for run in runs:
        save_curve(run, curves_dir)
        save_curve_csv(run, curve_csv_dir)

    write_summary(runs, outroot / "summary.csv")
    save_overlay(runs, outroot / "overlay_val_mse.png", metric="val_mse")
    save_overlay(runs, outroot / "overlay_train_mse.png", metric="train_mse")
    save_heatmaps(runs, heat_dir)

    print(f"Done. Processed {len(runs)} runs.")
    print(f"Outputs written under: {outroot}")
    print(f"- curves:    {curves_dir}")
    print(f"- curve_csv: {curve_csv_dir}")
    print(f"- heatmaps:  {heat_dir}")
    print(f"- summary:   {outroot / 'summary.csv'}")
    print(f"- overlays:  {outroot / 'overlay_val_mse.png'}, {outroot / 'overlay_train_mse.png'}")


if __name__ == "__main__":
    main()

