# xai-channel-flow User Guide

> For applied engineers who want to reproduce the SHAP/coherent-structure analysis on their own turbulent-channel DNS data.

---

## 1. What xai-channel-flow can do

- Train a convolutional U-Net to predict future turbulent-channel velocity fields from DNS snapshots.
- Compute per-grid-point SHAP (Expected Gradients) importance maps for that trained model.
- Detect classical coherent structures (Q events, streaks, Chong/Hunt vortices) and SHAP-thresholded structures via percolation analysis.
- Quantify the spatial coincidence between SHAP-important regions and each classical structure family.

## 2. Installation

No install matrix or packaged distribution exists — this is a script collection run in place from `code/`. No `requirements.txt`/`environment.yml` exists either; dependencies (TensorFlow/Keras, NumPy, h5py) must be inferred from imports (see [`developer_guide.md` §2 Metadata snapshot](./developer_guide.md#2-metadata-snapshot)).

```bash
git clone <this-repo>
cd xai-channel-flow/code
# install tensorflow, numpy, h5py yourself — no pinned versions provided
```

## 3. Choosing a backend / runtime

> Skipped — single-framework (TensorFlow/Keras); the only runtime choice is local vs. remote (SSH) data staging, covered in `developer_guide.md` §6.

## 4. Global configuration

Configuration is a set of plain Python modules under `code/configuration/`, imported by each `main_*.py` via `exec("from configuration import <name> as <alias>")`. Edit these files directly before running a stage — this mirrors the repo's own `README.md` step 1.

```python
# code/configuration/training_data.py (excerpt)
learat      = 5e-5     # learning rate
batch_size  = 1
field_ini   = 26000    # first DNS snapshot index used
field_fin   = 26010    # last DNS snapshot index used
nfil        = 1        # first-layer filter count of the U-Net
epoch_max   = 1
```

Key knobs:
- `code/configuration/channel_data.py` — channel geometry and physical parameters (`L_x`, `L_z`, `L_y`, `rey`, `utau`).
- `code/configuration/folders.py` — every input/output path and filename used by the pipeline (see `developer_guide.md` §4 for the full data flow this drives).

## 5. Core-object map

```
data/phys/*.h5.uvw
   -> main_statistics.py        (Umean, Urms, norm)
   -> prepare_tfrecords.py      (TFRecords for training)
   -> main_CNN.py                (trained_model.h5)
   -> main_SHAP.py               (*.h5.shap maps)
   -> create_SHAPstruc_dataset.py (*.struc)
   -> main_percolation_shapstruc_range.py (perc_*.txt)
   -> calc_*_coinc*.py            (*_coin.txt)
   -> plot_*.py                   (figures in results/plots/)
```

## 6. Flow domain definition

The domain is a fixed periodic turbulent channel, not a user-authored geometry. `code/configuration/channel_data.py` sets its size (`L_x = 8*pi`, `L_z = 3*pi`, `L_y = 1`) and friction Reynolds number/velocity (`rey`, `utau`); `code/py_bin/py_class/flow_field.py::flow_field` reads a DNS snapshot against this geometry.

## 7. Boundary / initial conditions

> Skipped — this repo consumes pre-computed DNS snapshots (`data/phys/*.h5.uvw`); it does not author or solve boundary/initial conditions itself.

## 8. Data objects

| Class | When to use |
| --- | --- |
| `flow_field` (`code/py_bin/py_class/flow_field.py`) | Load and reshape one DNS snapshot. |
| `uv_structure` / `streak_structure` / `chong_structure` / `hunt_structure` (`code/py_bin/py_class/`) | Detect one classical coherent-structure family from a snapshot. |
| `shap_structure` (+ `shap_uvw_structure`, `shap_uw_vsign_structure`, `shap_intensity_structure`) | Threshold a SHAP map into SHAP-based structures for coincidence comparison. |

## 9. PDE residual / constraint authoring

> Skipped — no PDE is solved by this code; the DNS input is treated as ground truth, not derived from a residual.

## 10. Networks

```python
# code/py_bin/py_class/ann_config.py (usage, from main_CNN.py)
import py_bin.py_class.ann_config as ann
Unet = ann.deep_model(DL_data)       # DL_data: paths, shapes, padding
Unet.define_model(Training_data)     # Training_data: learat, batch_size, nfil, ...
Unet.train_model()
```

Built-in: a single configurable U-Net (`deep_model.architecture_Unet`), no catalog of alternative architectures.

## 11. Training

```python
# code/main_CNN.py end-to-end (abridged)
exec("from configuration import training_data as tr_data")
Unet = ann.deep_model(DL_data)
Unet.define_model(Training_data)
Unet.train_model()
```

Key options (all in `code/configuration/training_data.py`): `learat`, `batch_size`, `epoch_max`, `epoch_save`, `read_model` (resume vs. fresh), `multi_worker`/`flag_central` (distributed strategy, see `developer_guide.md` §14).

## 12. Inverse problems

> Skipped — not applicable; this repo does forward prediction + post-hoc explanation, not parameter inversion.

## 13. Operator learning

> Skipped — not applicable.

## 14. Uncertainty quantification & multifidelity

Partial equivalent only: `code/configuration/shap_data.py::nrep_field` repeats the SHAP calculation over multiple stochastic repetitions per field so results can be averaged/checked for noise (`code/shap_check_repetitions.py`, `code/plot_shaps_noise.py`), which is a variance-control mechanism rather than full UQ.

## 15. Persistence

```python
# code/py_bin/py_class/ann_config.py::deep_model
# model_write / model_read (code/configuration/folders.py) name the .h5 checkpoint
# code/py_bin/py_class/shap_config.py::shap_config.write_shap / read_shap persist SHAP maps
```

## 16. Visualization & post-processing

`code/py_bin/py_plots/` backs roughly 30 `plot_*.py`/`calc_*.py` entry points in `code/` — see the full, individually-verified inventory in `topology.md` §3/§6 (module: `code`). Representative examples: `plot_training_epoch.py` (reads `hist_file`), `plot_predictions.py`/`calc_pred_error.py` (model evaluation), `plotstruc3d.py`/`plot_shap_3d.py` (3D structure rendering), `plotpercolation.py` (percolation curves), and the coincidence-analysis family (`calc_coinc*.py`/`plot_coinc*.py`) comparing SHAP-important regions against classical coherent structures (Q events, streaks, Chong vortices).

## 17. Parallel training

See `developer_guide.md` §14 — `multi_worker`/`flag_central` in `code/configuration/training_data.py`.

## 18. Troubleshooting

| Symptom | Suggestion |
| --- | --- |
| HDF5 file-locking errors on a shared/cluster filesystem | Already mitigated: every `main_*.py` sets `os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'` before touching h5py. |
| `main_CNN.py` fails to find TFRecords | Run `code/prepare_tfrecords.py` first when `code/configuration/training_data.py::prep_data = True`. |
| Not runtime-tested — this pass is a static code read; no GPU/data was available to execute the pipeline and surface additional failure modes. | — |

## 19. Minimal runnable templates

### 19.1 Statistics → training (the repo's own steps 1–4)

```bash
cd code
python main_statistics.py       # Umean.txt, Urms.txt, norm.txt
python prepare_tfrecords.py     # TFRecords for training (if prep_data=True)
python main_CNN.py              # trained_model.h5, hist.txt
```

### 19.2 SHAP → structures → coincidence (steps 6–14)

```bash
cd code
python main_SHAP.py                          # *.h5.shap maps
python create_SHAPstruc_dataset.py            # SHAP-thresholded structures
python main_percolation_shapstruc_range.py    # percolation curves
python calc_streak_shap_coinc_y_range.py      # example: streak/SHAP coincidence
```

## 20. Further reading

- Developer guide: [`developer_guide.md`](./developer_guide.md)
- Tutorial: [`tutorial.md`](./tutorial.md)
- Official docs: none — see `README.md` §Citation for the paper this pipeline reproduces.
