# eegecg_cca  (placeholder name)

Generic engine for EEG-ECG CCA analyses. Datasets enter through `WindowDataset` (one subject x condition x window length: aligned `meta`, `psd`, `ecg` tables);
the engine builds band features from the PSD, standardises inside training folds only, fits a one-component CCA (model registry: `cca`, plus any model you
`register_model`), and produces the evidence: purged blocked cross-validation, circular-shift permutation null, Haufe patterns and their stability,
leave-one-block-out, optional lag sweep.

**No analysis choices live in this package.** Bands, targets, QC, number of folds / permutations, seeds and the model are passed in through `CCAConfig`
(no defaults); model hyper-parameters (e.g. the ridge strengths of rCCA) are bound by the caller when registering the model.
The data readers for a specific store layout live in the dataset's analysis notebook.

Status: under construction.
