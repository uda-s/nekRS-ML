# FCN prhs to gradp

This package trains a simple element-wise fully connected network for the
`doe_tgv_gnn_offline` research case. It predicts `gradp` from `prhs` without
using graph connectivity.

The data loader pairs files by `(time, rank, size)` parsed from names such as:

```text
fld_prhs_time_5000_rank_0_size_8.bin
fld_gradp_time_5000_rank_0_size_8.bin
```

This avoids the Dist-GNN filename sorting issue while that code path is handled
separately.

## Quick Start

From a case or test directory:

```bash
python /mnt/data1/uda/nekRS-ML/3rd_party/fcn-prhs-gradp/train_fcn.py \
  --config /mnt/data1/uda/nekRS-ML/3rd_party/fcn-prhs-gradp/config.yaml \
  --set data.gnn_outputs_path=./gnn_outputs_poly_3_5-25 \
  --set output.output_dir=./fcn_outputs/default
```

For a fast smoke test:

```bash
python /mnt/data1/uda/nekRS-ML/3rd_party/fcn-prhs-gradp/train_fcn.py \
  --config /mnt/data1/uda/nekRS-ML/3rd_party/fcn-prhs-gradp/config.yaml \
  --set data.gnn_outputs_path=./gnn_outputs_poly_3_5-25 \
  --set data.max_samples=20000 \
  --set learning_rate_schedule.phase1_steps=10 \
  --set learning_rate_schedule.phase2_steps=10 \
  --set output.output_dir=./fcn_outputs/smoke
```

Each run writes:

- `resolved_config.yaml`
- `loss_history.csv`
- `loss_history.png`
- `best.pt`
- `last.pt`

Post-process a grid run with:

```bash
cd /mnt/data1/uda/Codex-test_tgv_gnn_offline
./post_fcn.sh
```

This writes per-run curves, curve CSV files, summary CSV, overlays, and heatmaps
under `fcn_outputs_grid/post/`.

For the local SLURM workflow, use the generator in the Codex test case:

```bash
cd /mnt/data1/uda/Codex-test_tgv_gnn_offline
./gen_runscript_fcn_slurm
sbatch run_fcn_grid.sbatch
```
