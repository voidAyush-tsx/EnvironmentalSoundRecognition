# Reconstructing the selected CNN14 model

- **Official backbone implementation:** https://github.com/qiuqiangkong/audioset_tagging_cnn ; pinned commit `d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4`.
- **Official pretrained AudioSet checkpoint:** https://zenodo.org/records/3576403 ; original filename `Cnn14_mAP=0.431.pth`; published/verified MD5 `595633ac2d1cac7ef04ebf70e2fee4e4`.
- **Project's restricted-loadable backbone file:** `checkpoints/Cnn14_AudioSet_state_dict_safe.pth` (exported from the verified checkpoint; normally omitted from Git/submission because of size).
- **Selected downstream checkpoint:** `results/stage4/20261027_144004_603447_block6_test1/best.pt` on the student's original PC, SHA-256 `7c0e01a0078c225a18f4dc5eaa5110d051ef2c8effec7bcebb4e856e6659328c`. This contains *trainable parameters only*, not the full backbone. The submission packager includes it under `models/selected_block6_best.pt` when present and verified.
- **Restore sequence:** obtain official CNN14 source and original weights, generate/use the safe backbone state dict, build the `block6`-adapted ten-class model from project `src/` code, load the selected trainable parameters, then run `src.evaluate_selected` on the official dataset. Follow Notebook 02 and its evaluation script, rather than loading `best.pt` as a complete CNN14 model.
- **Security:** `torch.load(..., weights_only=False)` should only ever be used with a verified trusted official legacy checkpoint. Our exported safe weights use restricted loading. Do not execute untrusted checkpoint pickles.
- **Dataset:** https://urbansounddataset.weebly.com/urbansound8k.html . Original WAV files are not placed into the ZIP; obtain separately and extract to `data/UrbanSound8K/`. Respect dataset terms.

The chosen fold-1 test score was 88.55% (873 examples), not a ten-fold mean. The evidence files contain the selected checkpoint SHA-256 and source commit for audit.
