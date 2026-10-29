from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA

from .data import FeatureGrid, WindowDataset
from .features import build_eeg_features
from .targets import build_targets
from .config import CCAConfig


def optional_qc_mask(ecg: pd.DataFrame, qc: dict) -> tuple[np.ndarray, dict]:
    """Optional, explicitly configured quality rules (all disabled when the setting is None)."""
    ok = np.ones(len(ecg), dtype=bool)
    counts = {}
    if qc.get("max_rejected_fraction") is not None:
        r = ecg["rejected_fraction"].to_numpy(dtype=float)
        rule = np.isfinite(r) & (r <= qc["max_rejected_fraction"])
        counts["n_fail_max_rejected_fraction"] = int((~rule).sum()); ok &= rule
    if qc.get("bpm_range") is not None:
        lo, hi = qc["bpm_range"]; b = ecg["bpm"].to_numpy(dtype=float)
        rule = np.isfinite(b) & (b >= lo) & (b <= hi)
        counts["n_fail_bpm_range"] = int((~rule).sum()); ok &= rule
    if qc.get("min_beats") is not None:
        rule = ecg["n_beats"].to_numpy(dtype=float) >= qc["min_beats"]
        counts["n_fail_min_beats"] = int((~rule).sum()); ok &= rule
    return ok, counts

def complete_case_mask(X: np.ndarray, Y: np.ndarray, ecg: pd.DataFrame, cfg: CCAConfig) -> tuple[np.ndarray, dict]:
    """Rows usable by CCA = finite EEG & finite targets (& optional QC). Counts are per-criterion (they overlap)."""
    eeg_ok = np.isfinite(X).all(axis=1)
    tgt_ok = np.isfinite(Y)
    qc_ok, qc_counts = optional_qc_mask(ecg, cfg.qc)
    keep = eeg_ok & tgt_ok.all(axis=1) & qc_ok
    report = {"n_windows_total": int(len(X)), "n_nonfinite_eeg": int((~eeg_ok).sum()),
              **{f"n_missing_{t}": int((~tgt_ok[:, j]).sum()) for j, t in enumerate(cfg.ecg_targets)},
              **qc_counts, "n_complete_windows": int(keep.sum())}
    return keep, report

@dataclass
class PreparedUnit:
    X: np.ndarray
    Y: np.ndarray
    meta: pd.DataFrame
    ecg: pd.DataFrame
    grid: FeatureGrid
    report: dict

def prepare_unit(ds: WindowDataset, cfg: CCAConfig) -> PreparedUnit:
    """Features -> targets -> complete-case filtering. Row order is preserved."""
    X, grid = build_eeg_features(ds, cfg)
    Y = build_targets(ds.ecg, cfg.ecg_targets)
    meta, ecg = ds.meta, ds.ecg
    keep, report = complete_case_mask(X, Y, ecg, cfg)
    return PreparedUnit(X[keep], Y[keep], meta[keep].reset_index(drop=True), ecg[keep].reset_index(drop=True), grid, report)

def min_windows_required(n_eeg_features: int, cfg: CCAConfig) -> int:
    return int(np.ceil(cfg.min_windows_factor * n_eeg_features))

def min_train_required(n_x: int, n_y: int, cfg: CCAConfig) -> float:
    """Unregularised CCA needs n_train > factor * max(n_X, n_Y) + 1."""
    return cfg.min_train_factor * max(n_x, n_y) + 1
