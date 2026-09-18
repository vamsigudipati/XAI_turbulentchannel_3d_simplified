# xai-channel-flow — From Beginner to Expert Tutorial

> Ten chapters, progressively deeper. This repo is a fixed 15-step numbered pipeline (see its own `README.md`), not a general-purpose library, so several chapters from the canonical outline (`references/doc_outlines.md`) don't map cleanly onto it — those are marked **skipped** below with the substitute topic actually covered, per that outline's own allowance to adjust numbering when a phase is clearly inapplicable.

## Table of contents

- Chapter 1 — Environment setup & Hello example (statistics stage)
- Chapter 2 — Domain basics (channel geometry)
- Chapter 3 — *(skipped: no constraint-authoring API)* → Data preparation (TFRecords)
- Chapter 4 — *(skipped: no PDE time-dependence)* → The CNN's temporal-prediction setup
- Chapter 5 — Convergence techniques (training hyperparameters)
- Chapter 6 — *(skipped: no parameter inversion)* → Computing SHAP attributions
- Chapter 7 — *(skipped: fixed channel geometry)* → Structure taxonomy (Q/streak/chong/hunt/SHAP)
- Chapter 8 — *(skipped: no operator learning)* → Percolation & coincidence analysis
- Chapter 9 — Acceleration (multi-worker / cluster)
- Chapter 10 — Mastery

---

## Chapter 1 · Environment setup & Hello example

See [`user_guide.md` §2 Installation](./user_guide.md#2-installation) and [§19 Minimal runnable templates](./user_guide.md#19-minimal-runnable-templates) for the same commands in reference form. Install: no packaged install exists; no dependency versions are pinned upstream either (see [`developer_guide.md` §2 Metadata snapshot](./developer_guide.md#2-metadata-snapshot)).

```bash
cd code
python main_statistics.py
```

This is adapted from `code/main_statistics.py`, the smallest first stage: it reads the 11 sample DNS snapshots in `data/phys/`, and writes `Umean.txt`, `Urms.txt`, `norm.txt` under `results/sta/` (paths from `code/configuration/folders.py`).

Expected result: three new text files under `results/sta/`, one per statistic.

**Exercise** — Open `code/configuration/stats_data.py` and change `field_fin` from `26010` to `26005`, so only 6 of the 11 sample fields are used; re-run and confirm the output files still write successfully.

---

## Chapter 2 · Domain basics

The channel geometry is fixed, not user-authored: `code/configuration/channel_data.py` sets `L_x = 8*pi`, `L_z = 3*pi`, `L_y = 1`, `rey ≈ 125.95`, `utau ≈ 0.05998`. `code/py_bin/py_class/flow_field.py::flow_field.__init__` consumes these values plus a folder/file pair to load one snapshot; `shape_tensor()` derives the downsampled tensor shape used by every later stage.

Representative example: `code/main_statistics.py` (constructs a `flow_field` and calls `shape_tensor()`).

**Exercise** — Read `code/configuration/channel_data.py::dx`, `dy`, `dz` and predict how changing `dx` from `1` to `2` would change the shape returned by `flow_field.shape_tensor()`.

---

## Chapter 3 · Data preparation (TFRecords)

Before training, `code/prepare_tfrecords.py` (module docstring names it `check_tfrecords.py`) converts raw `data/phys/*.h5.uvw` snapshots into TensorFlow's TFRecord format, using the same `ann.deep_model` class as training, but calling `Unet.prepare_tfrecords()` instead of `train_model()`. This step is required whenever `code/configuration/training_data.py::prep_data = True`.

**Exercise** — Trace `code/configuration/folders.py::tfrecord_folder` from `prepare_tfrecords.py` through to `main_CNN.py`; confirm both scripts reference the same folder.

---

## Chapter 4 · The CNN's temporal-prediction setup

`code/configuration/training_data.py::delta_pred` sets how many fields ahead the U-Net predicts (a fixed-horizon forecast, not a continuous-time PDE). `deep_model.pred_field(data_in={"index_ii":...})` (`code/py_bin/py_class/ann_config.py:892`) evaluates this at a given snapshot index.

**Exercise** — Find where `delta_pred` is consumed inside `ann_config.py` and note whether increasing it would require re-training or only re-evaluation.

---

## Chapter 5 · Convergence techniques

- Learning rate / momentum: `code/configuration/training_data.py::learat` (5e-5), `optmom` (0.9) — RMSprop, fixed, no schedule.
- Epoch budget: `epoch_max`, `epoch_save` (checkpoint cadence via `deep_model._save_training`).
- No adaptive sampling, no Adam→L-BFGS pipeline, no explicit loss-weighting exists in this repo — single fixed training configuration per run.

**Exercise** — Change `epoch_max` in `training_data.py` from `1` to `5` and confirm `results/sta/hist.txt` grows to 5 rows.

---

## Chapter 6 · Computing SHAP attributions

Example: `code/main_SHAP.py`. It loads the trained model (`folders.model_read`), builds a `shap_config` instance, and calls `calc_gradientSHAP()` — Expected Gradients (Erion et al., *Nat. Mach. Intell.* 3(7), 620–631, 2021, per the script's own docstring) via the vendored `shap.GradientExplainer`. Output: one `*.h5.shap` file per field under `data/SHAP/`.

```python
# code/main_SHAP.py (abridged)
shap_model = sc.shap_config(data_in=data_shap)
shap_model.calc_gradientSHAP()
```

The five ingredients of a SHAP run here:
1. A trained model (`folders.model_read`, from Chapter 4/5's training run).
2. The field range to explain (`code/configuration/shap_data.py::field_ini`/`field_fin`).
3. Sample count for Expected Gradients (`shap_data.py::nsamples`).
4. Repetition count for noise-averaging (`shap_data.py::nrep_field`).
5. Output location (`folders.shap_folder`/`shap_file`).

**Exercise** — Run `code/shap_check_repetitions.py` after setting `nrep_field > 1` and inspect `file_snr` (signal-to-noise ratio of the repeated SHAP estimate).

---

## Chapter 7 · Structure taxonomy (Q / streak / chong / hunt / SHAP)

Five structure-detection classes share one method surface (`calculate_matstruc`, `segment_struc`, `save_struc`, `read_struc`, `add_SHAP`):
- `uv_structure` — Q events (Reynolds-stress quadrants).
- `streak_structure` — velocity streaks.
- `chong_structure` / `hunt_structure` — two vortex-identification criteria.
- `shap_structure` (and its `_uvw`/`_uw_vsign`/`_intensity` variants) — SHAP-value-thresholded structures.

Pointers into the repo:
- `code/py_bin/py_class/chong_structure.py` — smallest example of the shared pattern.
- `code/create_SHAPstruc_dataset.py` — the entry point that builds `shap_structure` instances from a `*.h5.shap` file.

**Exercise** — Compare `chong_structure.py::calculate_matstruc` and `shap_structure.py::calculate_matstruc` and note what threshold parameter (`Hperc`, `filvol` from `channel_data.py`) each uses.

---

## Chapter 8 · Percolation & coincidence analysis

### 8.1 Percolation

Example: `code/main_percolation_shapstruc_range.py`, calling `code/py_bin/py_functions/percolation.py::percolation` over a range of thresholds (`stats_data_shap.py::Hmin`/`Hmax`/`Hnum`) to produce a percolation curve, saved via `save_percolation`.

### 8.2 Coincidence

Example: `code/calc_streak_shap_coinc_y_range.py`, calling `code/py_bin/py_functions/calc_coinc.py::calc_coinc` to quantify the fraction of a SHAP structure's volume that overlaps a classical structure's volume, as a function of wall distance — the core quantitative result of the underlying paper.

**Exercise** — Run `main_percolation_shapstruc_range.py` then one `calc_*_coinc_y_range.py` script on the same field range and confirm both read the same `folders.shap_folder`/`SHAPq_folder` outputs.

---

## Chapter 9 · Acceleration (multi-worker / cluster)

`code/main_CNN.py`'s module docstring documents the exact SLURM launch script used on the Chalmers "Alvis" cluster (4 nodes × 4 A100 GPUs, `srun`). `code/configuration/training_data.py::multi_worker`/`flag_central` select TensorFlow's `MirroredStrategy` vs. `CentralStorageStrategy`; `code/py_bin/py_functions/multiworker_checkpoint.py` handles checkpointing in the multi-worker case.

**Exercise** — Read the SLURM header in `main_CNN.py`'s docstring and identify which line sets the GPU count per node.

---

## Chapter 10 · Mastery

Pick any of the following extension points and implement a small patch:

1. Custom plotting script — follow `code/py_bin/py_plots/plotpercolation.py` as a template for a new `plot_*.py` entry point.
2. Custom structure type — subclass the pattern from Chapter 7 (see `developer_guide.md` §15.2 for the exact steps).
3. Custom coincidence metric — extend `code/py_bin/py_functions/calc_coinc.py` following `calc_coinc_type`/`calc_coinc_4struc` as examples of adding a new comparison shape.
4. Custom loss/metric — the current loss is fixed inside `ann_config.py::create_model` as `tf.keras.losses.MeanSquaredError()` (not a swappable identifier); see `developer_guide.md` §13 for the exact call sites to change.
5. Custom backend (advanced) — not applicable; this repo is TensorFlow-only with no backend-abstraction layer (see `developer_guide.md` §6).

**Exercise** — Choose one extension point above and open a PR that includes a new `code/configuration/*.py` entry if your extension needs a new tunable parameter.

---

## After this tutorial

- Deep source reading: [`developer_guide.md`](./developer_guide.md).
- API lookup: [`user_guide.md`](./user_guide.md).
- Official docs: none — see `README.md` §Citation for the paper this pipeline reproduces.
