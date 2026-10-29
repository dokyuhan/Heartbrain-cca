import numpy as np
import pandas as pd

def purged_blocked_splits(meta: pd.DataFrame, n_splits: int) -> list[dict]:
    """Returns [{train_idx, test_idx, n_purged}] per fold."""
    n = len(meta)
    idx = np.arange(n)
    fold_sizes = np.full(n_splits, n // n_splits, dtype=int)
    fold_sizes[: n % n_splits] += 1
    blocks = meta["block_id"].to_numpy()
    starts, ends = meta["start_sample"].to_numpy(), meta["end_sample"].to_numpy()

    splits, cur = [], 0
    for size in fold_sizes:
        test_idx = idx[cur: cur + size]
        train_idx = np.concatenate([idx[:cur], idx[cur + size:]])
        t_blocks, t_starts, t_ends = blocks[test_idx], starts[test_idx], ends[test_idx]
        keep = np.ones(len(train_idx), dtype=bool)
        for i, ti in enumerate(train_idx):
            same = t_blocks == blocks[ti]
            if same.any() and (same & (starts[ti] < t_ends) & (ends[ti] > t_starts)).any():
                keep[i] = False
        splits.append({"train_idx": train_idx[keep], "test_idx": test_idx, "n_purged": int((~keep).sum())})
        cur += size
    return splits
