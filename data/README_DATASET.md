# UrbanSound8K data (not committed)

Obtain the original UrbanSound8K dataset from https://urbansounddataset.weebly.com/urbansound8k.html and preserve its predefined folds.

Expected local layout:

```text
data/UrbanSound8K/metadata/UrbanSound8K.csv
data/UrbanSound8K/audio/fold1/*.wav
...
data/UrbanSound8K/audio/fold10/*.wav
```

The project uses all 8,732 official metadata entries and class IDs 0–9. Resample decoded waveforms to mono 32 kHz, then center-crop/right-pad to 128,000 samples; CNN14 computes the log-Mel features internally.

Raw WAV files and the original archive are deliberately absent from Git and from the faculty submission ZIP. See [reproducibility](../docs/REPRODUCIBILITY.md). Five `fsID` values are shared across official folds; see `results/cross_fold_source_summary.csv` in the original run or the faculty report for that limitation.

Dataset: Salamon, Jacoby & Bello, *A Dataset and Taxonomy for Urban Sound Research*, ACM Multimedia (2014). Follow original dataset license and attribution.
