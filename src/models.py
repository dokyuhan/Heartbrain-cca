import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from .data import _safe_corr

# Ordinary CCA
def fit_cca_1comp(Xz, Yz):
    model = CCA(
        n_components=1,
        scale=False,
        max_iter=1000
    ).fit(Xz, Yz)

    U, V = model.transform(Xz, Yz)

    return {
        "wx": model.x_weights_[:, 0].copy(),
        "wy": model.y_weights_[:, 0].copy(),
        "U": U[:, 0],
        "V": V[:, 0],
        "transform": lambda X, Y: tuple(
            a[:, 0] for a in model.transform(X, Y)
        )
    }

# L2-Regularized CCA
def inv_sqrt(matrix):
    """Inverse square root of a positive-definite matrix."""
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, 1e-12)
    return (vectors * (1.0 / np.sqrt(values))) @ vectors.T


def fit_rcca_1comp(Xz, Yz, alpha_x, alpha_y):
    """
    One-component ridge-regularized CCA.

    Xz: standardized EEG features, shape (n, 45)
    Yz: standardized ECG features, shape (n, 2)
    """
    mx = Xz.mean(axis=0)
    my = Yz.mean(axis=0)

    X = Xz - mx
    Y = Yz - my

    n = X.shape[0]

    # Sample covariance matrices
    Sxx = X.T @ X / (n - 1)
    Syy = Y.T @ Y / (n - 1)
    Sxy = X.T @ Y / (n - 1)

    # Ridge regularization
    Sxx += alpha_x * np.eye(X.shape[1])
    Syy += alpha_y * np.eye(Y.shape[1])

    # Whitened cross-covariance
    Wx = inv_sqrt(Sxx)
    Wy = inv_sqrt(Syy)

    K = Wx @ Sxy @ Wy

    # Leading canonical component
    U_svd, _, Vt = np.linalg.svd(K, full_matrices=False)

    wx = Wx @ U_svd[:, 0]
    wy = Wy @ Vt.T[:, 0]

    # Normalize training scores
    wx /= np.std(X @ wx)
    wy /= np.std(Y @ wy)

    U = X @ wx
    V = Y @ wy

    def transform(X_new, Y_new):
        return (X_new - mx) @ wx, (Y_new - my) @ wy

    return {
        "wx": wx,
        "wy": wy,
        "U": U,
        "V": V,
        "transform": transform
    }

# Registry of model fitters; a regularised model would be registered here (hyper-parameters tuned inside training folds only).
# Registry of model fitters. Only parameter-free models are registered here; a model with hyper-parameters (the ridge strengths of rCCA) is
# registered by the CALLING notebook with its own values:  register_model("rcca", functools.partial(fit_rcca_1comp, alpha_x=..., alpha_y=...)).
# Hyper-parameters must be fixed beforehand or tuned inside the training folds only.
MODEL_FITTERS = {"cca": fit_cca_1comp}

def register_model(name: str, fitter) -> None:
    """Adds (or replaces) a model: fitter(Xz, Yz) -> {"wx", "wy", "U", "V", "transform"} on standardised data."""
    MODEL_FITTERS[name] = fitter

def haufe_from_scores(Z: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Forward pattern  A = Cov(Z, score) / Var(score)  (Haufe et al. 2014). Z = standardised TRAINING data,
    score = the matching TRAINING canonical score. Z may be the EEG block (score U) or the cardiac block (score V)."""
    s = score.reshape(-1, 1)
    return ((Z.T @ (s - s.mean())) / Z.shape[0] / (s.var(ddof=0) + 1e-12)).ravel()

def orientation_sign(V_train: np.ndarray, y_anchor_z_train: np.ndarray) -> float:
    """ONE deterministic physiological orientation: the TRAINING cardiac variate V must correlate positively with the
    standardised training anchor (ln_rmssd by default). Applied once; nothing re-flips folds afterwards."""
    return -1.0 if _safe_corr(V_train, y_anchor_z_train) < 0 else 1.0

def oriented_component(fit: dict, Xz: np.ndarray, Yz: np.ndarray, anchor_idx: int) -> dict:
    """Apply the orientation to EVERYTHING together (both scores, both weight vectors) and derive the interpretation
    quantities from the TRAINING data: EEG Haufe pattern, cardiac weights / loadings / Haufe pattern."""
    s = orientation_sign(fit["V"], Yz[:, anchor_idx])
    U, V = s * fit["U"], s * fit["V"]
    return {
        "orientation_flipped": bool(s < 0), "U": U, "V": V,
        "eeg_weights": s * fit["wx"], "ecg_weights": s * fit["wy"],
        "eeg_haufe": haufe_from_scores(Xz, U),                                            # Cov(Xz, U) / Var(U)
        "ecg_loadings": np.array([_safe_corr(Yz[:, j], V) for j in range(Yz.shape[1])]),  # corr(Yz_j, V)
        "ecg_haufe": haufe_from_scores(Yz, V),                                            # Cov(Yz, V) / Var(V)
        "train_cca_r": _safe_corr(U, V),
        "anchor_corr_train": _safe_corr(V, Yz[:, anchor_idx]),                            # >= 0 by construction
    }

def fit_fold_model(X_tr, Y_tr, X_te, Y_te, model_type: str, anchor_idx: int, details: bool = True) -> dict:
    """One train/test fold. LEAKAGE RULES: both scalers and the model are fitted on the training rows only; the test rows
    are only transformed. The held-out r is SIGNED: corr(EEG score, cardiac score) on test data, never abs().
    details=False (permutation loop) runs the identical scaling/fit/score steps but skips the interpretation work."""
    sx, sy = StandardScaler().fit(X_tr), StandardScaler().fit(Y_tr)
    Xtr, Ytr, Xte, Yte = sx.transform(X_tr), sy.transform(Y_tr), sx.transform(X_te), sy.transform(Y_te)
    try:
        fit = MODEL_FITTERS[model_type](Xtr, Ytr)
        Ute, Vte = fit["transform"](Xte, Yte)
    except Exception as exc:                                    # e.g. a constant column; reported, never silent
        return {"fold_test_cca_r": np.nan, "error": repr(exc)}
    out = {"fold_test_cca_r": _safe_corr(Ute, Vte)}            # orientation flips U and V together: r is unaffected
    if details:
        out.update(oriented_component(fit, Xtr, Ytr, anchor_idx))
    return out


def fit_cca_full(X: np.ndarray, Y: np.ndarray, model_type: str, anchor_idx: int) -> dict:
    """DIAGNOSTIC / REFERENCE ONLY: fit on every complete window. full_train_cca_r is optimistic."""
    Xz, Yz = StandardScaler().fit_transform(X), StandardScaler().fit_transform(Y)
    comp = oriented_component(MODEL_FITTERS[model_type](Xz, Yz), Xz, Yz, anchor_idx)
    return {"full_train_cca_r": comp["train_cca_r"], "full_fit_eeg_haufe": comp["eeg_haufe"],
            "full_fit_ecg_loadings": comp["ecg_loadings"]}
