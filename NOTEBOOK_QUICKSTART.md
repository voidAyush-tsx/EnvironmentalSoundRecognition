# Notebook-first usage (Windows / VS Code)

Extract this kit into `D:\EnvironmentalSoundRecognition`, preserving `notebooks/` and replacing
root `README.md` and `.gitignore`. **Do not delete `src/`, `tests/`, `data/`, `checkpoints/`,
`results/` or the original official PANNs source.** The notebook expects the Stage 3/4 source
files already installed and audited in the preceding work.

1. In VS Code use **File → Open Folder → D:\EnvironmentalSoundRecognition**.
2. Open `notebooks/01_UrbanSound8K_PANNs_CNN14_Project.ipynb`.
3. Select the Python interpreter containing PyTorch `2.7.0+cu128`, e.g. the interpreter
   shown in earlier Windows checks: `C:\Program Files\Python313\python.exe`.
4. Read the first `USER CONTROLS` cell. Run all with opt-in flags **False** to inspect
   saved evidence and generate statistics/figures from the local dataset.
5. Press **Ctrl+S** to persist cell output in `.ipynb`. The executed notebook can be
   committed to Git; long-running logs are saved separately under `results/notebook_evidence/`.
6. To rerun the gradient verification: set `RUN_REAL_STAGE4_GRADIENT_CHECKS=True` and
   execute its cell. To start limited diagnostic training, separately enable
   `RUN_DIAGNOSTIC_PILOTS=True`. Full-fold training is **off** until enabled.
7. To reset Python bytecode caches, enable `CLEAR_PYTHON_CACHES=True` and run only the
   cleanup cell, then **Restart Kernel**. Do not remove saved results or checkpoints.

**.gitignore** ignores `__pycache__/`, `.ipynb_checkpoints/`, local dataset, checkpoints,
large assets and results. It does not delete them; it prevents untracked versions being pushed.
A file already staged/tracked by Git remains tracked despite `.gitignore`.
Check what Git will commit using `git status`, from a notebook `subprocess` cell if preferred.
