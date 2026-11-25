# NeuralNet CLI Usage

How to build and run the `NeuralNet` executable: what each mode does, what it prints, and which files it creates.

---

## 1. Build first

Running an exe never rebuilds it. Build the configuration you are about to run:

```
cmake --build out/build/x64-Release
```

- **Release** for any real run (training, benchmarks, logs).
- **Debug** only for stepping through code with breakpoints (20–50× slower).

If `out/build/x64-Release` doesn't exist yet, select **x64-Release** once in the Visual Studio dropdown (or run `cmake --preset x64-release`).

## 2. Running

From the project root:

```
.\out\build\x64-Release\NeuralNet.exe <mode> [flags]
```

or `cd out\build\x64-Release` once and use `.\NeuralNet.exe <mode> [flags]`.

Where you run it from doesn't matter: all data/model/log paths are absolute, set by CMake.

---

## 3. Modes

| Mode | What it does | Reads | Writes |
|---|---|---|---|
| `train` *(default)* | Trains on 80% of `train.csv`, validates on the other 20% each epoch | `data/train.csv` | log files, model `.bin` |
| `train-full` | Trains on all 42,000 images, no validation | `data/train.csv` | log files, model `<name>_full.bin` |
| `eval` | Loads a saved model and scores it on the validation split. No training | `data/train.csv`, model `.bin` | nothing |
| `predict` | Loads a saved model, predicts every image in `test.csv`, writes the Kaggle submission | `data/test.csv`, model `.bin` | `submissions/<name>_submission.csv` |

Running with no mode is the same as `train`.

### `train`

Terminal output:

```
logging run: 20261113-203300_cpu_naive (Release)
training on 33600 images, validating on 8400
epoch 1  lr=0.5  train_loss=0.78  val_loss=0.35  val_acc=89.5%  (4.7 s)
...
epoch 25 ...
saved model: C:/.../models/mlp_784-128-64-10_sigmoid_v1.bin
```

Files created:

- `logs/<run_id>_epochs.csv` — one row per epoch: `epoch, lr, train_loss, val_loss, val_acc, train_secs, images_per_sec, gflops`
- `logs/runs.csv` — one summary row appended per run (build type, backend, final/best accuracy, mean s/epoch, …). Never overwritten.
- `models/<model_name>.bin` — trained weights. **Overwrites** any existing model with the same name.

### `train-full`

Same as `train`, except:

- No validation set, so epoch lines show only `train_loss` and time; `val_loss` / `val_acc` are logged as `nan`.
- Model is saved as `models/<model_name>_full.bin`, so it doesn't overwrite the 80% model.
- Backend label in the logs gets `_full` added.

### `eval`

Terminal output:

```
model mlp_784-128-64-10_sigmoid_v1 on 8400 validation images:  acc=97.11%  loss=0.0987
```

- Uses the **same** 8,400 validation images as `train` (fixed split seed), so the score is fair.
- Should match the last epoch's `val_acc` of the run that saved the model. A mismatch means a save/load problem.
- Evaluating a `_full` model this way is misleading: it trained on those images.

### `predict`

Terminal output:

```
wrote 28000 predictions to C:/.../submissions/mlp_784-128-64-10_sigmoid_v1_submission.csv
```

File format (Kaggle Digit Recognizer):

```
ImageId,Label
1,2
2,0
...
```

Upload it on the competition page via **Submit Predictions**.

---

## 4. Flags

Flags work in any order, before or after the mode.

| Flag | Effect | Applies to |
|---|---|---|
| `--no-log` | Don't create any log files (console output unchanged) | `train`, `train-full` |
| `--no-save` | Don't write the model `.bin` | `train`, `train-full` |
| `--model NAME` | Use model file `models/NAME.bin` instead of the default. No `.bin` in `NAME` | all modes |

Default model name is `model_name` in `TrainConfig` (`include/config.hpp`) (`_full` added for `train-full`).

---

## 5. Common recipes

**Quick test run, leave no files behind**

```
.\NeuralNet.exe train --no-log --no-save
```

**Train, then check the saved model**

```
.\NeuralNet.exe train
.\NeuralNet.exe eval
```

**Kaggle submission from the 80% model**

```
.\NeuralNet.exe predict
```

**Kaggle submission from a full-data model**

```
.\NeuralNet.exe train-full
.\NeuralNet.exe predict --model mlp_784-128-64-10_sigmoid_v1_full
```

**Keep an older model instead of overwriting it**

```
.\NeuralNet.exe train --model mlp_784-128-64-10_sigmoid_v2
.\NeuralNet.exe eval  --model mlp_784-128-64-10_sigmoid_v2
```

---

## 6. Where things go

All relative to the project root (set in `CMakeLists.txt`):

| Folder | Contents | In git? |
|---|---|---|
| `data/` | `train.csv`, `test.csv` (inputs) | no |
| `models/` | trained `.bin` weight files | no |
| `logs/` | per-run epoch CSVs, `runs.csv` | no |
| `submissions/` | Kaggle CSVs | no |

Folders are created automatically when first needed.

---

## 7. Changing settings

| Setting | Where | Needs rebuild? |
|---|---|---|
| Mode, logging, saving, model name | command-line flags | no |
| Epochs, batch size, learning rate, decay, split, seeds | `TrainConfig` in `include/config.hpp` | **yes** |
| `backend` label (for benchmark logs) | `TrainConfig` in `include/config.hpp` | **yes** |
| Logging / saving defaults | `Options` defaults in `include/cli.hpp` | **yes** |

---

## 8. Errors you might see

| Message | Cause | Fix |
|---|---|---|
| `unknown argument 'X'` + usage line | Typo, or the exe is an old build | Check spelling; rebuild |
| `load: could not open .../models/NAME.bin` | No model with that name yet | Run `train` (without `--no-save`) first, or check `--model` |
| `load: file has a (AxB) matrix, network expects (CxD)` | Model saved from a different architecture | Use the matching model / layer sizes |
| `mnist::load: expected 784 pixels ... got 783` | Labelled loader pointed at `test.csv` | Only `predict` reads `test.csv` |
| Every mode trains | Stale exe from before the CLI was added | `cmake --build` the folder you run from |
