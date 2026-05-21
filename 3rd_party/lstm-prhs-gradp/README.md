# LSTM prhs to gradp

This package trains an element-wise LSTM for the `doe_tgv_gnn_offline`
research case. It predicts `gradp` from a short temporal sequence of `prhs`
without using graph connectivity.

Each sample is one node over `sequence_length` consecutive snapshots. The
target is `gradp` at the last timestep in the sequence.

## Quick Start

```bash
python /mnt/data1/uda/nekRS-ML/3rd_party/lstm-prhs-gradp/train_lstm.py \
  --config /mnt/data1/uda/nekRS-ML/3rd_party/lstm-prhs-gradp/config.yaml \
  --set data.gnn_outputs_path=./gnn_outputs_poly_3_5-25 \
  --set output.output_dir=./lstm_outputs/default
```

For the local SLURM workflow:

```bash
cd /mnt/data1/uda/Codex-test_tgv_gnn_offline
./gen_runscript_lstm_slurm
sbatch run_lstm_grid.sbatch
./post_lstm.sh
```

