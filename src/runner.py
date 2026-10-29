import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA

from .data import WindowDataset, _cosine
from .targets import resolve_orientation_anchor
from .qc import min_windows_required, prepare_unit
from .splits import purged_blocked_splits
from .models import fit_cca_full
from .validation import aggregate_cv_patterns, cross_validate_cca, leave_one_block_out_cca, permutation_null_pvalue
from .config import CCAConfig


def run_cca_unit(ds: WindowDataset, cfg: CCAConfig, verbose: bool = True) -> dict:
    assert cfg.n_components == 1, "only a one-component CCA is implemented in this version"
    anchor_idx, anchor_name = resolve_orientation_anchor(cfg.ecg_targets, cfg.orientation_target)
    prep = prepare_unit(ds, cfg)
    grid, targets = prep.grid, cfg.ecg_targets
    key = {"subject": ds.subject_id, "condition": ds.condition_id, "window_length": ds.window_length,
           "ecg_target_set": cfg.ecg_target_set}
    blocks = list(pd.unique(prep.meta["block_id"])) if len(prep.meta) else []
    base = {**key, "ecg_targets": json.dumps(targets), "orientation_anchor": anchor_name, "blocks": ",".join(blocks),
            "n_windows": len(prep.meta), "n_eeg_features": grid.n_features, "n_ecg_features": len(targets),
            **{k: v for k, v in prep.report.items() if k != "n_complete_windows"}}
    out = {"summary": [], "folds": [], "features": [], "ecg_patterns": [], "loro": [],
           "qc_report": [{**key, "optional_qc": json.dumps(cfg.qc), **prep.report}]}
    if verbose:
        print(f"  {ds.subject_id} / {ds.condition_id} / {ds.window_length}s / [{cfg.ecg_target_set}]: "
              f"{prep.report['n_complete_windows']}/{prep.report['n_windows_total']} complete windows "
              f"(non-finite EEG {prep.report['n_nonfinite_eeg']}; missing "
              + ", ".join(f"{t} {prep.report[f'n_missing_{t}']}" for t in targets) + f"); blocks={blocks}")

    need = min_windows_required(grid.n_features, cfg)
    if len(prep.meta) < need:
        out["summary"].append({**base, "status": "insufficient_windows", "n_folds_used": 0, "n_purged_total": 0})
        if verbose: print(f"      -> SKIPPED (n_windows={len(prep.meta)} < {need})")
        return out
    X, Y, meta = prep.X, prep.Y, prep.meta

    full = fit_cca_full(X, Y, cfg.model_type, anchor_idx)                        
    splits = purged_blocked_splits(meta, cfg.n_splits)
    cv = cross_validate_cca(X, Y, meta, cfg, anchor_idx, details=True, splits=splits)      
    perm = permutation_null_pvalue(X, Y, meta, cfg, anchor_idx, cfg.n_perm, splits, observed_cv_r=cv["cv_cca_r"])

    row = {**base, "status": "ok" if cv["n_folds_used"] else "no_valid_folds",
           "full_train_cca_r": full["full_train_cca_r"], "cv_cca_r": cv["cv_cca_r"],
           "cv_mean_r": cv.get("cv_mean_r"), "cv_std_r": cv.get("cv_std_r"),
           "cv_min_r": cv.get("cv_min_r"), "cv_max_r": cv.get("cv_max_r"),
           "n_folds_used": cv["n_folds_used"], "n_folds_skipped": cv["n_folds_skipped"],
           "skipped_folds": json.dumps(cv["skipped_folds"]), "n_purged_total": cv["n_purged_total"],
           "permutation_p": perm["p_value"], "null_mean": perm["null_mean"], "null_std": perm["null_std"],
           "null_median": perm["null_median"], "null_95th_percentile": perm["null_95th_percentile"],
           "n_null_ge_observed": perm["n_null_ge_observed"], "n_perm": perm["n_perm_used"]}

    if cv["folds"]:
        agg = aggregate_cv_patterns(cv["folds"], grid, targets)
        row.update(agg["summary"]); row.update(agg["similarity"])
        row["n_orientation_flips"] = int(sum(f["orientation_flipped"] for f in cv["folds"]))
        row["n_folds_negative_cosine_with_mean"] = agg["n_folds_negative_cosine_with_mean"]
        row["full_vs_cv_haufe_cosine"] = _cosine(full["full_fit_eeg_haufe"], agg["mean_unit_pattern"])   # signed
        for j, t in enumerate(targets):
            r = agg["ecg_patterns"].iloc[j]
            row.update({f"ecg_loading_{t}_mean": r["loading_mean"], f"ecg_haufe_{t}_mean": r["haufe_mean"],
                        f"ecg_weight_{t}_mean": r["weight_mean"], f"full_fit_ecg_loading_{t}": full["full_fit_ecg_loadings"][j]})
        out["features"] += [{**key, "feature": n, **v} for n, v in agg["feature"].to_dict("index").items()]
        out["ecg_patterns"] += [{**key, **r} for r in agg["ecg_patterns"].to_dict("records")]
        for j, f in enumerate(cv["folds"]):
            A = agg["eeg_haufe_raw"][j]
            fr = {**key, "fold": f["fold"], "n_train": f["n_train"], "n_test": f["n_test"], "n_purged": f["n_purged"],
                  "fold_train_cca_r": f["train_cca_r"], "fold_test_cca_r": f["fold_test_cca_r"],
                  "orientation_flipped": f["orientation_flipped"], "orientation_anchor": anchor_name}
            fr.update({f"ecg_weight_{t}": float(f["ecg_weights"][i]) for i, t in enumerate(targets)})
            fr.update({f"ecg_loading_{t}": float(f["ecg_loadings"][i]) for i, t in enumerate(targets)})
            fr.update({f"ecg_haufe_{t}": float(f["ecg_haufe"][i]) for i, t in enumerate(targets)})
            fr.update({f"haufe_{c}": float(v) for c, v in zip(grid.names, A)})
            fr.update({f"weight_{c}": float(v) for c, v in zip(grid.names, f["eeg_weights"])})
            out["folds"].append(fr)

    if cfg.run_loro:                                                                  
        loro = leave_one_block_out_cca(X, Y, meta, cfg, anchor_idx)
        row.update({k: loro[k] for k in ("loro_median_r", "loro_mean_r", "loro_std_r", "n_loro_blocks")})
        out["loro"] += [{**key, **r} for r in loro["per_block"]]
    out["summary"].append(row)
    if verbose and np.isfinite(row["cv_cca_r"]):
        print(f"      -> full_train_r={row['full_train_cca_r']:.3f}  cv_cca_r={row['cv_cca_r']:+.3f} "
              f"(folds used {row['n_folds_used']}, skipped {row['n_folds_skipped']})  perm_p={row['permutation_p']}")
    return out

RESULT_KEYS = ["summary", "folds", "features", "ecg_patterns", "loro", "qc_report"]

def run_cca_analysis(datasets, cfg: CCAConfig) -> dict:
    """Generic top-level runner: ANY iterable of WindowDatasets (Colemak now, visuomotor later) -> result DataFrames."""
    acc = {k: [] for k in RESULT_KEYS}
    for ds in datasets:
        res = run_cca_unit(ds, cfg)
        for k in acc:
            acc[k].extend(res[k])
    return {k: pd.DataFrame(v) for k, v in acc.items()}

SUMMARY_LEAD = (["subject", "condition", "window_length", "ecg_target_set", "ecg_targets", "status", "blocks",
                 "n_windows", "n_windows_total", "n_eeg_features", "n_ecg_features", "orientation_anchor",
                 "full_train_cca_r",                                                     
                 "cv_cca_r", "cv_mean_r", "cv_std_r", "cv_min_r", "cv_max_r", "n_folds_used", "n_folds_skipped",
                 "n_purged_total", "permutation_p", "null_mean", "null_std", "null_median", "null_95th_percentile",
                 "n_null_ge_observed", "n_perm", "loro_median_r", "loro_mean_r", "loro_std_r", "n_loro_blocks",
                 "haufe_cosine_mean", "haufe_cosine_median", "haufe_cosine_min", "haufe_abs_cosine_mean",
                 "haufe_pearson_mean", "n_orientation_flips", "n_folds_negative_cosine_with_mean", "full_vs_cv_haufe_cosine"])

def order_summary(df: pd.DataFrame) -> pd.DataFrame:
    lead = [c for c in SUMMARY_LEAD if c in df.columns]
    return df[lead + [c for c in df.columns if c not in lead]]

def export_results(out: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {"cca_summary.csv": order_summary(out["summary"]), "cca_fold_results.csv": out["folds"],
             "cca_haufe_features.csv": out["features"], "cca_ecg_patterns.csv": out["ecg_patterns"],
             "cca_loro_results.csv": out["loro"], "cca_window_report.csv": out["qc_report"]}
    for name, df in files.items():
        df.to_csv(output_dir / name, index=False)
        print(f"Saved {len(df):5d} rows -> {output_dir / name}")
