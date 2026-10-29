import numpy as np
import pandas as pd

def build_targets(ecg: pd.DataFrame, ecg_targets: list[str]) -> np.ndarray:
    """Y = ecg[ecg_targets] -> [n_windows, k], k >= 1. Values are read as stored (ln_rmssd is NOT recomputed)."""
    missing = [t for t in ecg_targets if t not in ecg.columns]
    if missing:
        raise KeyError(f"ECG target(s) {missing} not found in the dataset's ecg table (available: {list(ecg.columns)})")
    return ecg[ecg_targets].to_numpy(dtype=float)

def resolve_orientation_anchor(ecg_targets: list[str], orientation_target: str) -> tuple[int, str]:
    """Which Y column fixes the sign of the cardiac canonical variate. Falls back explicitly to the first target."""
    if orientation_target in ecg_targets:
        return ecg_targets.index(orientation_target), orientation_target
    print(f"  [orientation] '{orientation_target}' is not in ECG targets {ecg_targets}; "
          f"using the first target '{ecg_targets[0]}' as the sign anchor")
    return 0, ecg_targets[0]
