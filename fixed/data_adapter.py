from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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
            import api  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "루트의 api 모듈을 불러오지 못했습니다. app_state.py와 Manatee 데이터가 "
                "기존 실행 환경에서 로드 가능한지 확인하세요."
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

