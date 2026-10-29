from dataclasses import dataclass, field
import numpy as np
import pandas as pd

@dataclass
class WindowDataset:
    """The ONE internal representation every dataset adapter must return (for one subject x condition x window length).
    Rows of `meta`, `psd` and `ecg` are aligned and sorted by (block_id, window_index).
      meta  : subject_id, condition_id, block_id, window_length, window_index, start_sample, end_sample, start_time, end_time
      psd   : [n_windows, n_channels, n_frequencies]   (power spectral density, as stored by the extraction)
      frequency_hz : [n_frequencies]      channels : list of channel names (from the store, in PSD order)
      ecg   : DataFrame with every cached cardiac column (ln_rmssd, bpm, ... + quality diagnostics), same rows as meta
      info  : free-form dict (provenance, blocks present, ...)  -- the engine never depends on it."""
    meta: pd.DataFrame
    psd: np.ndarray
    frequency_hz: np.ndarray
    channels: list
    ecg: pd.DataFrame
    info: dict = field(default_factory=dict)

    @property
    def subject_id(self): return str(self.meta["subject_id"].iloc[0])
    @property
    def condition_id(self): return str(self.meta["condition_id"].iloc[0])
    @property
    def window_length(self): return int(self.meta["window_length"].iloc[0])

META_COLUMNS = ["subject_id", "condition_id", "block_id", "window_length", "window_index",
                "start_sample", "end_sample", "start_time", "end_time"]

def validate_window_dataset(ds: WindowDataset) -> None:
    """Contract check for any adapter's output."""
    n = len(ds.meta)
    missing = [c for c in META_COLUMNS if c not in ds.meta.columns]
    assert not missing, f"meta is missing columns {missing}"
    assert ds.psd.ndim == 3 and ds.psd.shape[0] == n and ds.psd.shape[1] == len(ds.channels), \
        f"psd shape {ds.psd.shape} vs n={n}, channels={len(ds.channels)}"
    assert ds.psd.shape[2] == len(ds.frequency_hz), "psd.shape[2] != len(frequency_hz)"
    assert len(ds.ecg) == n, "ecg table not aligned with meta"
    assert ds.meta["window_length"].nunique() == 1, "a WindowDataset must hold ONE window length (15 s and 30 s never mix)"
    keys = list(zip(ds.meta["block_id"], ds.meta["window_index"]))
    assert keys == sorted(keys), "rows must be sorted by (block_id, window_index)"

@dataclass
class FeatureGrid:
    """Layout of the EEG feature matrix: channel-major, i.e. column = channel_index * n_bands + band_index."""
    channels: list
    bands: list

    @property
    def names(self): return [f"{ch}_{b}" for ch in self.channels for b in self.bands]
    @property
    def n_features(self): return len(self.channels) * len(self.bands)

def _safe_corr(a, b) -> float:
    a, b = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def _cosine(a, b) -> float:
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else np.nan

def _nan_row_dict(keys): return {k: np.nan for k in keys}
