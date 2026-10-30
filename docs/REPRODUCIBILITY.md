# Reproducibility and fresh-PC setup

## Requirements

- Python 3.13; verified on Windows with PyTorch `2.7.0+cu128` and NVIDIA RTX 3050 (4 GiB VRAM).
- See `requirements.txt` and `requirements.lock.txt` for direct project dependencies. The exact lock reflects the original Windows environment.
- Git, VS Code/Jupyter recommended. A CUDA GPU is useful for training; CPU forward verification is supported.

## External assets are not tracked

1. Download the official [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) archive and extract into `data/UrbanSound8K/` (expect `metadata/UrbanSound8K.csv` and `audio/fold1` through `audio/fold10`).
2. Obtain the official [PANNs code](https://github.com/qiuqiangkong/audioset_tagging_cnn) at commit `d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4`. If the external folder is absent or empty, run:
   ```powershell
   git clone https://github.com/qiuqiangkong/audioset_tagging_cnn.git external/audioset_tagging_cnn
   git -C external/audioset_tagging_cnn checkout d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4
   ```
   Do not assume the placeholder Git entry alone installs the external library.
3. Download original `Cnn14_mAP=0.431.pth` from [official Zenodo](https://zenodo.org/records/3576403) into `checkpoints/`. Expected MD5: `595633ac2d1cac7ef04ebf70e2fee4e4`. The source is a **legacy pickle checkpoint**. Only load it with unrestricted pickle if you trust the source and have independently verified provenance. Use `src/check_cnn14.py` to export `checkpoints/Cnn14_AudioSet_state_dict_safe.pth` for later restricted loading.
4. The **selected trained parameters** are not in Git. The local faculty ZIP includes `models/selected_block6_best.pt` only after matching SHA-256 `7c0e01a0078c225a18f4dc5eaa5110d051ef2c8effec7bcebb4e856e6659328c`. Restore that same verified file for reproducing the held-out test.

## Install and validate

Run in the project root:

```powershell
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -s tests -v
python run_all_checks.py --quick
```

When starting from an empty checkout, verify the dataset, PANNs revision and checkpoint access **before** running the combined checker. The quick checker uses the tensor-only safe checkpoint and is not a downloader for 1.4 GB model weights.

## Running research notebooks

1. Open `notebooks/01_UrbanSound8K_PANNs_CNN14_Project.ipynb`. Check its user controls and start with costly operations disabled.
2. Open Notebook 02 to inspect existing comparison/test evidence. Its held-out evaluation is an **opt-in** operation. Do **not** use fold 1 for model selection.
3. Open Notebook 03 for CO4/CO5 and faculty audit.
4. Save notebooks with Ctrl+S to retain cell outputs. Running only the read-only cells is enough for a presentation.

## Experiments

Training configurations are `head`, `fc1` and `block6`. To rerun the original single-fold comparison:

```powershell
python -m src.finetune_train --config head --test-fold 1 --epochs 6 --patience 2 --batch-size 1 --accum-steps 4
python -m src.finetune_train --config fc1 --test-fold 1 --epochs 6 --patience 2 --batch-size 1 --accum-steps 4
python -m src.finetune_train --config block6 --test-fold 1 --epochs 6 --patience 2 --batch-size 1 --accum-steps 4
```

These create **new runs**, not bit-identical guarantee across hardware/software. Reproduction of the previously selected held-out result requires the matching selected trainable-parameters checkpoint and official CNN14 state dict; see [checkpoint access](../models/CHECKPOINT_ACCESS.md).

## Faculty audit

```powershell
python tools/faculty_submission.py audit --project .
```

The faculty's final ZIP is named by **registration number only**. The ZIP is separate from Git; confirm the current README, saved notebooks and attribution before submission.

## Limitations

One held-out test fold only; five cross-fold source IDs; external AudioSet pretraining; no training-from-scratch or 10-fold benchmark experiment. See [CO4](CO4_PUBLISHED_METHODS.md) and [CO5](CO5_ABLATION_AND_DISCUSSION.md).
