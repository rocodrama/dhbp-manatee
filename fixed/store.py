from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="microseconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class RunStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    source_stage TEXT NOT NULL,
                    target_stage TEXT NOT NULL,
                    models_json TEXT NOT NULL,
                    n_iter INTEGER NOT NULL,
                    n_llm_candidates INTEGER NOT NULL,
                    run_dir TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS iterations (
                    iteration_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    llm_reason TEXT,
                    best_strategy TEXT,
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    screening_csv TEXT,
                    deg_csv TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );

                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id TEXT PRIMARY KEY,
                    iteration_id TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    tf TEXT NOT NULL,
                    action TEXT NOT NULL,
                    FOREIGN KEY(iteration_id) REFERENCES iterations(iteration_id)
                );

                CREATE TABLE IF NOT EXISTS raw_llm_outputs (
                    raw_id TEXT PRIMARY KEY,
                    iteration_id TEXT NOT NULL,
                    raw_response TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(iteration_id) REFERENCES iterations(iteration_id)
                );

                CREATE TABLE IF NOT EXISTS screening_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    iteration_id TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    row_json TEXT NOT NULL,
                    FOREIGN KEY(iteration_id) REFERENCES iterations(iteration_id)
                );

                CREATE TABLE IF NOT EXISTS deg_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    iteration_id TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    gene TEXT NOT NULL,
                    row_json TEXT NOT NULL,
                    FOREIGN KEY(iteration_id) REFERENCES iterations(iteration_id)
                );
                """
            )

    def create_run(
        self,
        source_stage: str,
        target_stage: str,
        models: list[str],
        n_iter: int,
        n_llm_candidates: int,
        run_dir: Path,
    ) -> str:
        run_id = new_id("run")
        created_at = now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, source_stage, target_stage, models_json, n_iter,
                    n_llm_candidates, run_dir, status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?, ?)
                """,
                (
                    run_id,
                    source_stage,
                    target_stage,
                    json.dumps(models, ensure_ascii=False),
                    n_iter,
                    n_llm_candidates,
                    str(run_dir),
                    created_at,
                    created_at,
                ),
            )
        return run_id

    def finish_run(self, run_id: str, status: str = "completed") -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, updated_at = ? WHERE run_id = ?",
                (status, now_iso(), run_id),
            )

    def add_iteration(
        self,
        run_id: str,
        model_name: str,
        iteration: int,
        llm_reason: str | None = None,
        best_strategy: str | None = None,
        metrics: dict[str, Any] | None = None,
        screening_csv: str | None = None,
        deg_csv: str | None = None,
        error: str | None = None,
    ) -> str:
        iteration_id = new_id("iter")
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO iterations (
                    iteration_id, run_id, model_name, iteration, llm_reason,
                    best_strategy, metrics_json, screening_csv, deg_csv, error, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    iteration_id,
                    run_id,
                    model_name,
                    iteration,
                    llm_reason,
                    best_strategy,
                    json.dumps(metrics or {}, ensure_ascii=False, default=str),
                    screening_csv,
                    deg_csv,
                    error,
                    now_iso(),
                ),
            )
        return iteration_id

    def add_candidates(self, iteration_id: str, candidates: list[dict[str, str]]) -> None:
        rows = [
            (new_id("cand"), iteration_id, idx + 1, item["tf"], item["action"])
            for idx, item in enumerate(candidates)
        ]
        with self.connect() as conn:
            conn.executemany(
                "INSERT INTO candidates (candidate_id, iteration_id, rank, tf, action) VALUES (?, ?, ?, ?, ?)",
                rows,
            )

    def add_raw_output(self, iteration_id: str, raw_response: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO raw_llm_outputs (raw_id, iteration_id, raw_response, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (new_id("raw"), iteration_id, raw_response, now_iso()),
            )

    def add_dataframe_rows(self, table: str, iteration_id: str, df: pd.DataFrame, limit: int | None = None) -> None:
        if table not in {"screening_results", "deg_results"}:
            raise ValueError(f"지원하지 않는 테이블입니다: {table}")

        frame = df if limit is None else df.head(limit)
        rows = []
        for rank, (_, row) in enumerate(frame.iterrows(), start=1):
            row_dict = row.to_dict()
            if table == "deg_results":
                rows.append((iteration_id, rank, str(row_dict.get("gene", "")), json.dumps(row_dict, ensure_ascii=False, default=str)))
            else:
                rows.append((iteration_id, rank, json.dumps(row_dict, ensure_ascii=False, default=str)))

        with self.connect() as conn:
            if table == "deg_results":
                conn.executemany(
                    "INSERT INTO deg_results (iteration_id, rank, gene, row_json) VALUES (?, ?, ?, ?)",
                    rows,
                )
            else:
                conn.executemany(
                    "INSERT INTO screening_results (iteration_id, rank, row_json) VALUES (?, ?, ?)",
                    rows,
                )

