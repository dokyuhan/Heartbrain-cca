"""eegecg_cca (placeholder name): EEG-ECG canonical correlation engine with purged blocked CV, permutation null, Haufe patterns and a model registry.
Analysis choices are passed in through CCAConfig (no defaults); datasets enter through WindowDataset."""
from .config import CCAConfig
from .data import WindowDataset, validate_window_dataset, META_COLUMNS, FeatureGrid
from .models import (MODEL_FITTERS, register_model, fit_cca_1comp, fit_rcca_1comp, fit_fold_model, fit_cca_full,
                     haufe_from_scores, orientation_sign, oriented_component)
from .qc import prepare_unit, PreparedUnit
from .runner import run_cca_unit, run_cca_analysis, order_summary, export_results, RESULT_KEYS

__version__ = "1.0.0"
