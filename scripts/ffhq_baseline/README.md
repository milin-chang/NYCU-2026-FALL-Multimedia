# FFHQ baseline data preparation

`prepare.py` and `sr.py` are copied from the `314554035_Baseline` branch of
`milin-chang/NYCU-2026-FALL-Multimedia` (commit `49ef6617e19cf37b43933ba8e337c67feb8440fb`).
The two KAIR utility modules under `assets/upstream/kair/utils/` are from KAIR
commit `fc1732f4a4514e42ce15e5b3a1e18c828af47a1e`. The single compatibility
change to `utils_blindsr.py` replaces `scipy.finfo` with `np.finfo`, as the
upstream baseline's `prepare.py assets` command does.

From the repository root, after installing `kagglehub` in `src/.venv`, run:

```powershell
$env:KAGGLEHUB_CACHE = (Join-Path (Resolve-Path .\dataset).Path 'kaggle')
$env:MPLCONFIGDIR = (Join-Path (Resolve-Path .\src).Path '.mpl-cache')
& .\src\.venv\Scripts\python.exe .\scripts\ffhq_baseline\prepare.py dataset --out .\dataset\ffhq_baseline_1000_125_125 --counts 1000 125 125
```

KaggleHub downloads the full archive, extracts it under `dataset/kaggle/`,
and removes the archive after successful extraction.
The output is isolated from the project's existing FFHQ split and contains
`train/hr`, `val/{hr,lr}`, `test/{hr,lr}`, and `manifest.json`.

This baseline prepares data for the **real-world SR** task. It resizes FFHQ to
512×512 HR and uses KAIR's BSRGAN degradation to make 128×128 val/test LR.
The project's existing Lightweight SwinIR run uses bicubic ×4 LR and a
different 400/50/50 split, so scores from the two setups are not directly
comparable. The specified Kaggle version lists 52,001 PNGs, which is fewer
than the 70,000 images in full FFHQ.

The prepared manifest SHA-256 is
`faabb9e13efe0fcd7b3d5317c44157a14b69f8b63e05b16c5949131b8357eb1c`.
A copy of the manifest is committed at `spec/ffhq_baseline_1000_125_125_manifest.json`.
Run `python scripts/ffhq_baseline/verify.py` to check counts, dimensions, hashes,
and overlap with the project's older split. One image in the new test set,
`38081.png`, belongs to the older 400-image training split. Exclude it if
evaluating that already trained model as an unseen test sample.
