import numpy as np
from .data import FeatureGrid, WindowDataset
from .config import CCAConfig

def compute_band_features(psd: np.ndarray, frequency_hz: np.ndarray, bands: dict,
                          closed_upper_bands=()) -> np.ndarray:
    """Integrate the stored PSD over each band with the trapezoidal rule.
    psd [N, C, F], frequency_hz [F]  ->  band power [N, C, B]  (B = len(bands), in dict order).
    Band = PSD bins with lo <= f < hi, or lo <= f <= hi for bands named in `closed_upper_bands`.
    NaN PSD rows (flat/non-finite channel) propagate to NaN band power; a band with < 2 bins gives NaN."""
    f = np.asarray(frequency_hz, dtype=float)
    assert psd.ndim == 3 and psd.shape[2] == len(f), f"psd {psd.shape} vs {len(f)} frequencies"
    assert np.all(np.diff(f) > 0), "frequency_hz must be strictly increasing"
    tol = 1e-9
    out = np.full(psd.shape[:2] + (len(bands),), np.nan)
    for j, (name, (lo, hi)) in enumerate(bands.items()):
        if hi > f[-1] + tol or lo < f[0] - tol:
            raise ValueError(f"band {name}=({lo},{hi}) lies outside the stored PSD range [{f[0]}, {f[-1]}] Hz")
        upper = (f <= hi + tol) if name in closed_upper_bands else (f < hi - tol)
        mask = (f >= lo - tol) & upper
        if mask.sum() >= 2:
            out[:, :, j] = np.trapezoid(psd[:, :, mask], f[mask], axis=-1)
    return out

def build_eeg_features(ds: WindowDataset, cfg: CCAConfig) -> tuple[np.ndarray, FeatureGrid]:
    bp = compute_band_features(ds.psd, ds.frequency_hz, cfg.bands, cfg.closed_upper_bands)
    grid = FeatureGrid(channels=list(ds.channels), bands=list(cfg.bands))
    X = bp.reshape(len(bp), -1)                       # channel-major: matches FeatureGrid.names
    assert X.shape[1] == grid.n_features
    return X, grid
