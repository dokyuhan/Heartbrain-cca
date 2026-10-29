"""Analysis configuration. Every field is REQUIRED (no defaults): the analysis choices belong to the calling notebook, not to the package."""
from dataclasses import dataclass


@dataclass
class CCAConfig:
    bands: dict
    closed_upper_bands: tuple
    ecg_targets: list
    orientation_target: str
    n_components: int
    n_splits: int
    min_windows_factor: float
    min_train_factor: float
    min_test_windows: int
    qc: dict
    n_perm: int
    perm_seed: int
    run_loro: bool
    min_test_windows_loro: int
    model_type: str

    @property
    def ecg_target_set(self) -> str:
        return "+".join(self.ecg_targets)
