import numpy as np
import pandas as pd

from .data import FeatureGrid, WindowDataset, _cosine, _nan_row_dict
from .qc import min_train_required, min_windows_required, prepare_unit
from .splits import purged_blocked_splits
from .models import fit_fold_model
from .config import CCAConfig

STABILITY_EPS = 1e-9   # numerical guard in the Haufe stability summaries (not a study parameter)

def cross_validate_cca(X, Y, meta, cfg: CCAConfig, anchor_idx: int, details: bool = True, splits=None) -> dict:
    if splits is None:
        splits = purged_blocked_splits(meta, cfg.n_splits)
    need_train = min_train_required(X.shape[1], Y.shape[1], cfg)
    folds, fold_corrs, skipped, n_purged_total = [], [], [], 0
    for k, sp in enumerate(splits, start=1):
        tr, te = sp["train_idx"], sp["test_idx"]
        n_purged_total += sp["n_purged"]
        if len(tr) <= need_train:
            skipped.append({"fold": k, "reason": f"n_train={len(tr)} <= required {need_train:g}"}); continue
        if len(te) < cfg.min_test_windows:
            skipped.append({"fold": k, "reason": f"n_test={len(te)} < {cfg.min_test_windows}"}); continue
        res = fit_fold_model(X[tr], Y[tr], X[te], Y[te], cfg.model_type, anchor_idx, details)
        if not np.isfinite(res["fold_test_cca_r"]):
            skipped.append({"fold": k, "reason": res.get("error", "undefined test correlation")}); continue
        fold_corrs.append(float(res["fold_test_cca_r"]))
        if details:
            folds.append({"fold": k, "n_train": int(len(tr)), "n_test": int(len(te)), "n_purged": int(sp["n_purged"]), **res})
    arr = np.array(fold_corrs)
    out = {"cv_cca_r": float(np.median(arr)) if arr.size else np.nan, "fold_corrs": fold_corrs,
           "n_folds_used": len(fold_corrs), "n_folds_skipped": len(skipped), "skipped_folds": skipped,
           "n_purged_total": n_purged_total}
    if details:
        out.update({"cv_mean_r": float(arr.mean()) if arr.size else np.nan,
                    "cv_std_r": float(arr.std(ddof=1)) if arr.size > 1 else np.nan,
                    "cv_min_r": float(arr.min()) if arr.size else np.nan,
                    "cv_max_r": float(arr.max()) if arr.size else np.nan, "folds": folds})
    return out

def pattern_similarity(A_unit: np.ndarray) -> dict:
    """Pairwise similarity of the k ALREADY-ORIENTED unit-length fold patterns (rows). Diagnostics only: nothing is
    flipped to raise these numbers."""
    k = A_unit.shape[0]
    names = [f"haufe_{m}_{s}" for m in ("cosine", "abs_cosine", "pearson") for s in ("mean", "median", "min")]
    if k < 2:
        return _nan_row_dict(names)
    iu = np.triu_indices(k, 1)
    cos = (A_unit @ A_unit.T)[iu]
    pear = np.corrcoef(A_unit)[iu]
    out = {}
    for name, v in (("cosine", cos), ("abs_cosine", np.abs(cos)), ("pearson", pear)):
        out.update({f"haufe_{name}_mean": float(v.mean()), f"haufe_{name}_median": float(np.median(v)),
                    f"haufe_{name}_min": float(v.min())})
    return out

def summarize_haufe_by_grid(A_unit: np.ndarray, A_raw: np.ndarray, grid: FeatureGrid) -> dict:
    """Band- and channel-level summaries of the k oriented fold patterns (k, n_features).
    *_haufe_signed  : signed mean over the other axis, then over folds (DIRECTIONAL, unit-length patterns)
    *_haufe_absmag  : mean |pattern| over the other axis, then over folds (magnitude only; no direction)
    *_absmag_share  : absmag / sum over bands (or channels) -> sums to 1 within a unit
    *_haufe_abs_raw : magnitude on the raw (un-normalised) scale, for reference"""
    k, C, B = A_unit.shape[0], len(grid.channels), len(grid.bands)
    U3, R3 = A_unit.reshape(k, C, B), A_raw.reshape(k, C, B)
    sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
    out = {}
    for axis_name, labels, ax in (("band", grid.bands, 1), ("channel", grid.channels, 2)):
        signed, absmag, absraw = U3.mean(axis=ax), np.abs(U3).mean(axis=ax), np.abs(R3).mean(axis=ax)   # (k, n_labels)
        share = absmag.mean(axis=0) / absmag.mean(axis=0).sum()
        for j, lab in enumerate(labels):
            m, s = float(signed[:, j].mean()), sd(signed[:, j])
            out[f"{lab}_haufe_signed"] = m
            out[f"{lab}_haufe_signed_std"] = s
            out[f"{lab}_haufe_stability"] = abs(m) / (s + STABILITY_EPS) if np.isfinite(s) else np.nan
            out[f"{lab}_haufe_absmag"] = float(absmag[:, j].mean())
            out[f"{lab}_haufe_abs_raw"] = float(absraw[:, j].mean())
            out[f"{lab}_absmag_share"] = float(share[j])
    return out

def aggregate_cv_patterns(folds: list[dict], grid: FeatureGrid, ecg_targets: list[str]) -> dict:
    """Combine the per-fold (already oriented) training-data patterns. Nothing is re-aligned."""
    P_raw = np.vstack([f["eeg_haufe"] for f in folds])                       # (k, p)
    norms = np.linalg.norm(P_raw, axis=1, keepdims=True)
    P_unit = P_raw / np.where(norms == 0, 1.0, norms)
    k = len(folds)
    feat_std = P_unit.std(axis=0, ddof=1) if k > 1 else np.full(P_unit.shape[1], np.nan)
    feat_mean = P_unit.mean(axis=0)
    feature_df = pd.DataFrame({
        "mean_unit": feat_mean, "median_unit": np.median(P_unit, axis=0), "std_unit": feat_std,
        "stability": np.abs(feat_mean) / (feat_std + STABILITY_EPS),
        "mean_raw": P_raw.mean(axis=0), "median_raw": np.median(P_raw, axis=0),
        "std_raw": P_raw.std(axis=0, ddof=1) if k > 1 else np.nan}, index=grid.names)

    ecg_rows = []
    for j, t in enumerate(ecg_targets):
        row = {"ecg_target": t}
        for kind, key in (("weight", "ecg_weights"), ("loading", "ecg_loadings"), ("haufe", "ecg_haufe")):
            v = np.array([f[key][j] for f in folds])
            row.update({f"{kind}_mean": float(v.mean()), f"{kind}_median": float(np.median(v)),
                        f"{kind}_std": float(v.std(ddof=1)) if k > 1 else np.nan,
                        f"{kind}_n_positive_folds": int((v > 0).sum())})
        ecg_rows.append(row)
    mean_pat = P_unit.mean(axis=0)
    return {"eeg_haufe_raw": P_raw, "eeg_haufe_unit": P_unit, "feature": feature_df,
            "ecg_patterns": pd.DataFrame(ecg_rows), "summary": summarize_haufe_by_grid(P_unit, P_raw, grid),
            "similarity": pattern_similarity(P_unit), "mean_unit_pattern": mean_pat,
            "n_folds_negative_cosine_with_mean": int(sum(_cosine(p, mean_pat) < 0 for p in P_unit))}

def block_index_lists(meta: pd.DataFrame) -> list[np.ndarray]:
    """Row indices of each block in order of appearance (rows are sorted by block, then window)."""
    blocks = meta["block_id"].to_numpy()
    return [np.where(blocks == b)[0] for b in pd.unique(blocks)]

def permute_Y_within_blocks(Y: np.ndarray, block_idx: list[np.ndarray], rng: np.random.RandomState) -> np.ndarray:
    """One random circular shift per block, applied to ALL columns of Y at once."""
    Yp = Y.copy()
    for idx in block_idx:
        shift = rng.randint(1, max(2, len(idx) - 1))
        Yp[idx] = np.roll(Y[idx], shift, axis=0)
    return Yp

PERM_KEYS = ["p_value", "null_mean", "null_std", "null_median", "null_95th_percentile", "n_null_ge_observed"]

def permutation_null_pvalue(X, Y, meta, cfg: CCAConfig, anchor_idx: int, n_perm: int, splits=None,
                            observed_cv_r: float | None = None) -> dict:
    rng = np.random.RandomState(cfg.perm_seed)
    nan_out = {**_nan_row_dict(PERM_KEYS), "observed_cv_r": np.nan, "n_perm_used": 0}
    if splits is None:
        splits = purged_blocked_splits(meta, cfg.n_splits)
    observed = observed_cv_r if observed_cv_r is not None else \
        cross_validate_cca(X, Y, meta, cfg, anchor_idx, details=False, splits=splits)["cv_cca_r"]
    if n_perm <= 0 or not np.isfinite(observed):
        return {**nan_out, "observed_cv_r": observed}
    block_idx = block_index_lists(meta)
    null_vals = []
    for _ in range(n_perm):
        Yp = permute_Y_within_blocks(Y, block_idx, rng)
        val = cross_validate_cca(X, Yp, meta, cfg, anchor_idx, details=False, splits=splits)["cv_cca_r"]
        if np.isfinite(val):
            null_vals.append(val)
    if not null_vals:
        return {**nan_out, "observed_cv_r": observed}
    null = np.array(null_vals)
    n_ge = int(np.sum(null >= observed))
    return {"p_value": float((n_ge + 1) / (len(null) + 1)), "observed_cv_r": float(observed),
            "null_mean": float(null.mean()), "null_std": float(null.std(ddof=1)) if len(null) > 1 else np.nan,
            "null_median": float(np.median(null)), "null_95th_percentile": float(np.percentile(null, 95)),
            "n_null_ge_observed": n_ge, "n_perm_used": int(len(null))}

def leave_one_block_out_cca(X, Y, meta, cfg: CCAConfig, anchor_idx: int) -> dict:
    blocks = meta["block_id"].to_numpy()
    idx = np.arange(len(meta))
    need_train = min_train_required(X.shape[1], Y.shape[1], cfg)
    rows = []
    for b in pd.unique(blocks):
        te, tr = idx[blocks == b], idx[blocks != b]
        row = {"held_out_block": str(b), "n_train": int(len(tr)), "n_test": int(len(te)),
               "train_cca_r": np.nan, "test_cca_r": np.nan, "status": "ok"}
        if len(te) < cfg.min_test_windows_loro:
            row["status"] = "too_few_test_windows"
        elif len(tr) <= need_train:
            row["status"] = "too_few_train_windows"
        else:
            res = fit_fold_model(X[tr], Y[tr], X[te], Y[te], cfg.model_type, anchor_idx, details=True)
            row["test_cca_r"] = res["fold_test_cca_r"]
            row["train_cca_r"] = res.get("train_cca_r", np.nan)
            if not np.isfinite(row["test_cca_r"]):
                row["status"] = "undefined_correlation"
        rows.append(row)
    r = np.array([x["test_cca_r"] for x in rows if x["status"] == "ok"])
    return {"per_block": rows, "loro_median_r": float(np.median(r)) if r.size else np.nan,
            "loro_mean_r": float(r.mean()) if r.size else np.nan,
            "loro_std_r": float(r.std(ddof=1)) if r.size > 1 else np.nan, "n_loro_blocks": int(r.size)}
