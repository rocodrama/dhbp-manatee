from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
RESULTS_ROOT = REPO_ROOT / "results"


DEFAULT_MODELS = (
    "qwen2.5:14b",
)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RunConfig:
    source_stage: str = os.getenv("MANATEE_SOURCE_STAGE", "E5.25")
    target_stage: str = os.getenv("MANATEE_TARGET_STAGE", "E6.25")
    n_iter: int = _env_int("MANATEE_N_ITER", 3)
    n_llm_candidates: int = _env_int("MANATEE_N_LLM_CANDIDATES", 30)
    top_n_context: int = _env_int("MANATEE_TOP_N_CONTEXT", 20)
    top_n_trrust: int = _env_int("MANATEE_TOP_N_TRRUST", 25)
    top_n_feedback_genes: int = _env_int("MANATEE_TOP_N_FEEDBACK_GENES", 20)
    top_n_history: int = _env_int("MANATEE_TOP_N_HISTORY", 3)
    compute_mmd: bool = _env_bool("MANATEE_COMPUTE_MMD", False)
    use_all_tfs_as_allowed_list: bool = _env_bool("MANATEE_USE_ALL_TFS", True)
    models: tuple[str, ...] = field(default_factory=lambda: DEFAULT_MODELS)
    exhaustive_csv: Path = REPO_ROOT / "results" / "manatee_e525_to_e625_all_2tf_parallel" / "all_2tf_combinations_ranked.csv"
    results_root: Path = RESULTS_ROOT

    def make_run_dir(self) -> Path:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_transition = f"{self.source_stage}_to_{self.target_stage}".replace(".", "")
        return self.results_root / "main_refactor" / safe_transition / f"run_{run_id}"


CONFIG = RunConfig()
