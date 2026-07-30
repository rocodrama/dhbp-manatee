from __future__ import annotations

import sys
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent


def _candidate_api_roots() -> list[Path]:
    roots = []
    env_root = os.getenv("MANATEE_API_ROOT")
    if env_root:
        roots.append(Path(env_root).expanduser().resolve())
    roots.extend([REPO_ROOT, PACKAGE_ROOT])
    return _unique_existing_dirs(roots)


def _candidate_data_roots() -> list[Path]:
    roots = []
    env_root = os.getenv("MANATEE_DATA_ROOT")
    if env_root:
        roots.append(Path(env_root).expanduser().resolve())
    roots.extend([Path.cwd(), PACKAGE_ROOT, REPO_ROOT])
    return _unique_existing_dirs(roots)


def _unique_existing_dirs(paths: list[Path]) -> list[Path]:
    seen = set()
    out = []
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.exists() or not resolved.is_dir():
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _resolve_api_root() -> Path:
    for root in _candidate_api_roots():
        if (root / "api.py").exists():
            return root
    return REPO_ROOT


def _resolve_data_root() -> Path:
    for root in _candidate_data_roots():
        if (root / "Manatee").exists():
            return root
    return Path.cwd()


API_ROOT = _resolve_api_root()
DATA_ROOT = _resolve_data_root()
for path in (API_ROOT, REPO_ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@dataclass
class ManateeData:
    genes: Any
    tfs: Any
    labels: Any
    x: Any
    vae: Any
    trrust: dict[str, list[str]]

    @classmethod
    def from_api_module(cls) -> "ManateeData":
        try:
            with _working_directory(DATA_ROOT):
                import api  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "api 모듈을 불러오지 못했습니다. 서버 구조가 다르면 환경변수로 경로를 지정하세요.\n"
                f"- 감지된 api root: {API_ROOT}\n"
                f"- 감지된 data root: {DATA_ROOT}\n"
                f"- 현재 작업 디렉터리: {Path.cwd()}\n"
                "예: MANATEE_API_ROOT=/path/to/temp MANATEE_DATA_ROOT=/path/to/temp/main python -m main.run"
            ) from exc

        missing = [name for name in ("genes", "tfs", "labels", "x", "vae") if not hasattr(api, name)]
        if missing:
            raise RuntimeError(f"api 모듈에 필요한 속성이 없습니다: {', '.join(missing)}")

        return cls(
            genes=api.genes,
            tfs=api.tfs,
            labels=api.labels,
            x=api.x,
            vae=api.vae,
            trrust=getattr(api, "trrust", {}) or {},
        )

    def gene_name_to_idx(self) -> dict[str, int]:
        return {gene: idx for idx, gene in enumerate(self.genes)}

    def valid_tfs(self) -> list[str]:
        gene_idx = self.gene_name_to_idx()
        return [tf for tf in self.tfs if tf in gene_idx]

    def stage_indices(self, source_stage: str, target_stage: str) -> tuple[np.ndarray, np.ndarray]:
        source_idx = np.where(self.labels == source_stage)[0]
        target_idx = np.where(self.labels == target_stage)[0]
        if len(source_idx) == 0:
            raise ValueError(f"{source_stage} stage cell을 찾지 못했습니다.")
        if len(target_idx) == 0:
            raise ValueError(f"{target_stage} stage cell을 찾지 못했습니다.")
        return source_idx, target_idx

    def latent(self, arr: Any | None = None) -> np.ndarray:
        values = self.x if arr is None else arr
        with torch.no_grad():
            _, _, mu, _ = self.vae.forward(torch.FloatTensor(values))
        return mu.detach().cpu().numpy()
