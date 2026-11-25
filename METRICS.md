# Metrics

What each logged number means and how it's computed.

## Model quality

| Metric | Definition |
|---|---|
| `train_loss` | Mean cross-entropy over the epoch's mini-batches. Measured *while* weights change, so slightly optimistic early on. |
| `val_loss` | Cross-entropy on the held-out 20%, after the epoch. |
| `val_acc` | Fraction of validation images where `argmax(prediction) == label`. |
| `final_val_acc` / `best_val_acc` | Last and highest `val_acc` in a run (`runs.csv`). |

`train-full` has no validation set, so these are `nan`.

## Performance

| Metric | Definition |
|---|---|
| `train_secs` | Wall-clock time of one epoch's training loop (shuffle and validation excluded). |
| `images_per_sec` | `n_train / train_secs`. Training throughput. |
| **FLOPs** | A *count* of floating-point operations: how much work. Computed from layer shapes, not measured. |
| **FLOP/s** | FLOPs per second: how fast. `gflops` = GFLOP/s. |
| `gflops` | `train_flops_per_sample × n_train / train_secs / 1e9`. |
| **Peak** | Hardware's theoretical max: `cores × GHz × FLOPs/cycle/core`. |
| **MFU** (`mfu_pct`) | Model FLOPs Utilisation: `gflops / peak × 100`. How much of the hardware the code actually uses. |

## FLOP counting rules

- Matmul `(m×k)·(k×n)` = `2mkn` FLOPs (one multiply + one add per term).
- Only matmuls are counted ("model FLOPs"); activations, softmax and updates are <1% and ignored.
- Per image, per layer (`n_in → n_out`):
  - forward `W·a`: `2·n_in·n_out`
  - backward `dW = δ·aᵀ`: `2·n_in·n_out`
  - backward `Wᵀ·δ`: `2·n_in·n_out` (skipped for layer 0)
- 784-128-64-10: **218 kFLOP** forward, **454 kFLOP** train, **15.3 GFLOP**/epoch on 33,600 images.

FLOPs per epoch depend only on architecture and dataset size, never on batch size or code speed. Only time changes.

## Hardware peaks

| Device | Peak FP32 | How |
|---|---|---|
| Ryzen 7 5800X (chip) | 973 GFLOP/s | 8 cores × 3.8 GHz × 32 (2 FMA × 8 floats AVX2 × 2) |
| Ryzen 7 5800X (1 core) | 122 GFLOP/s | single-threaded code's realistic ceiling |
| RTX 2060 | ~6,500 GFLOP/s | vendor spec |

Base clock is used; boost (4.7 GHz) would raise the peak ~24%.

## Run-level averages (`runs.csv`)

`mean_gflops` and `mean_images_per_sec` are **total work / total time**, not the mean of per-epoch rates (which over-weights fast epochs).

## Comparing runs

- Same architecture, different code → compare `gflops` / `mfu_pct`.
- Different architecture → `train_secs` changes with the work; `gflops` shows whether efficiency changed.
- Only compare **Release** builds (`build` column).
