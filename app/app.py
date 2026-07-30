from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main.config import CONFIG, RunConfig


CSS = """
:root {
  --panel-border: #d8dde6;
  --panel-bg: #ffffff;
  --soft-bg: #f6f8fb;
  --text-muted: #5b6472;
}
body {
  background: var(--soft-bg);
}
.manatee-shell {
  max-width: 1440px;
  margin: 0 auto;
}
.manatee-title h1 {
  font-size: 22px;
  margin: 0 0 4px;
}
.manatee-title p {
  color: var(--text-muted);
  margin: 0 0 14px;
}
.trace-box textarea {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
}
"""


def _import_gradio():
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "gradio 패키지를 찾지 못했습니다. GUI를 쓰려면 기존 앱 실행 환경에서 실행하거나 gradio를 설치하세요."
        ) from exc
    return gr


def _parse_models(text: str) -> list[str]:
    models = [item.strip() for item in str(text or "").replace("\n", ",").split(",") if item.strip()]
    return models or list(CONFIG.models)


def _run_dirs() -> list[str]:
    root = CONFIG.results_root / "main_refactor"
    if not root.exists():
        return []
    dirs = [path.parent for path in root.rglob("manatee_run.sqlite3")]
    dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [str(path) for path in dirs]


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_sqlite_trace(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        return {"error": f"SQLite 파일이 없습니다: {db_path}"}

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        run_rows = [dict(row) for row in conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()]
        iteration_rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT iteration_id, model_name, iteration, llm_reason, best_strategy,
                       metrics_json, screening_csv, deg_csv, error, created_at
                FROM iterations
                ORDER BY model_name ASC, iteration ASC, created_at ASC
                """
            ).fetchall()
        ]
        candidate_rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT iteration_id, rank, tf, action
                FROM candidates
                ORDER BY iteration_id ASC, rank ASC
                """
            ).fetchall()
        ]
        raw_rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT iteration_id, created_at, raw_response
                FROM raw_llm_outputs
                ORDER BY created_at ASC
                """
            ).fetchall()
        ]

    for row in iteration_rows:
        try:
            row["metrics"] = json.loads(row.pop("metrics_json") or "{}")
        except json.JSONDecodeError:
            row["metrics"] = {}

    candidates_by_iteration: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        candidates_by_iteration.setdefault(str(row.get("iteration_id")), []).append(
            {
                "rank": row.get("rank"),
                "tf": row.get("tf"),
                "action": row.get("action"),
            }
        )

    screening_overview = []
    for row in iteration_rows:
        iteration_id = str(row.get("iteration_id"))
        candidates = candidates_by_iteration.get(iteration_id, [])
        candidate_tfs = [str(item.get("tf")) for item in candidates if item.get("tf")]
        unique_tf_pairs = []
        top_strategies = []
        evaluated_strategy_count = None
        screening_csv = row.get("screening_csv")
        if screening_csv:
            csv_path = Path(str(screening_csv))
            if csv_path.exists():
                try:
                    screen_df = pd.read_csv(csv_path)
                    evaluated_strategy_count = int(len(screen_df))
                    pair_keys = (
                        screen_df[["tf1", "tf2"]]
                        .astype(str)
                        .drop_duplicates()
                        .apply(lambda item: f"{item['tf1']} + {item['tf2']}", axis=1)
                        .tolist()
                    )
                    unique_tf_pairs = pair_keys
                    top_cols = [
                        col
                        for col in (
                            "strategy",
                            "distance_reduction_percent",
                            "mean_cell_improvement",
                            "mmd_similarity",
                            "target_correlation",
                            "cosine_direction",
                            "matches_both_llm_directions",
                        )
                        if col in screen_df.columns
                    ]
                    top_strategies = screen_df.head(50)[top_cols].to_dict("records") if top_cols else []
                except Exception as exc:
                    top_strategies = [{"error": f"screening CSV를 읽지 못했습니다: {exc}"}]

        tf_pair_count = int(len(candidate_tfs) * (len(candidate_tfs) - 1) / 2)
        screening_overview.append(
            {
                "iteration_id": iteration_id,
                "model_name": row.get("model_name"),
                "iteration": row.get("iteration"),
                "candidate_tf_count": len(candidate_tfs),
                "candidate_tfs": candidate_tfs,
                "candidate_perturbations": candidates,
                "unique_tf_pair_count": len(unique_tf_pairs) if unique_tf_pairs else tf_pair_count,
                "unique_tf_pairs": unique_tf_pairs,
                "evaluated_strategy_count": evaluated_strategy_count,
                "strategy_count_formula": "candidate_tf_count C 2 * 4 directions",
                "top_50_evaluated_strategies": top_strategies,
            }
        )

    raw_preview = []
    for row in raw_rows:
        raw_text = str(row.get("raw_response") or "")
        raw_preview.append(
            {
                "iteration_id": row.get("iteration_id"),
                "created_at": row.get("created_at"),
                "raw_response_preview": raw_text[:2000],
                "raw_response_length": len(raw_text),
            }
        )

    return {
        "runs": run_rows,
        "iterations": iteration_rows,
        "screening_overview": screening_overview,
        "candidates": candidate_rows,
        "raw_llm_outputs": raw_preview,
    }


def _load_summary(run_dir_text: str) -> pd.DataFrame:
    run_dir = Path(run_dir_text)
    summary_csv = run_dir / "iterative_summary.csv"
    if not summary_csv.exists():
        return pd.DataFrame()
    return pd.read_csv(summary_csv)


def load_run_trace(run_dir_text: str) -> tuple[pd.DataFrame, dict[str, Any], str]:
    if not run_dir_text:
        return pd.DataFrame(), {"message": "선택된 run이 없습니다."}, ""

    run_dir = Path(run_dir_text)
    trace = {
        "run_dir": str(run_dir),
        "sqlite": _read_sqlite_trace(run_dir / "manatee_run.sqlite3"),
        "iteration_history": _read_json(run_dir / "iteration_history.json"),
        "raw_llm_outputs": _read_json(run_dir / "raw_llm_outputs.json"),
        "final_answer": _read_json(run_dir / "final_answer.json"),
    }
    files = "\n".join(str(path) for path in sorted(run_dir.glob("*")))
    return _load_summary(run_dir_text), trace, files


def refresh_runs() -> Any:
    gr = _import_gradio()
    choices = _run_dirs()
    value = choices[0] if choices else None
    return gr.update(choices=choices, value=value)


def run_from_gui(
    query: str,
    source_stage: str,
    target_stage: str,
    model_text: str,
    n_iter: int,
    n_candidates: int,
    final_answer: bool,
) -> Any:
    yield (
        "실행을 시작했습니다. LLM 호출과 Manatee screening이 끝나면 trace가 표시됩니다.",
        pd.DataFrame(),
        {"status": "running"},
        "",
        None,
    )

    from main.agent.runner import default_query, run_workflow

    models = _parse_models(model_text)
    config = RunConfig(
        source_stage=source_stage,
        target_stage=target_stage,
        n_iter=int(n_iter),
        n_llm_candidates=int(n_candidates),
        top_n_context=CONFIG.top_n_context,
        top_n_trrust=CONFIG.top_n_trrust,
        top_n_feedback_genes=CONFIG.top_n_feedback_genes,
        top_n_history=CONFIG.top_n_history,
        use_all_tfs_as_allowed_list=CONFIG.use_all_tfs_as_allowed_list,
        models=tuple(models),
        exhaustive_csv=CONFIG.exhaustive_csv,
        results_root=CONFIG.results_root,
    )
    try:
        result = run_workflow(
            config=config,
            query=query or default_query(source_stage, target_stage),
            models=models,
            n_iter=int(n_iter),
            n_candidates=int(n_candidates),
            source_stage=source_stage,
            target_stage=target_stage,
            make_final_answer=bool(final_answer),
        )
        summary, trace, files = load_run_trace(result["run_dir"])
        yield "완료했습니다.", summary, trace, files, result["run_dir"]
    except Exception as exc:
        yield f"오류가 발생했습니다: {exc}", pd.DataFrame(), {"error": str(exc)}, "", None


def build_demo():
    gr = _import_gradio()
    run_choices = _run_dirs()
    default_models = ", ".join(CONFIG.models)

    with gr.Blocks(title="Manatee Trace GUI", css=CSS) as demo:
        gr.HTML(
            """
            <div class="manatee-shell manatee-title">
              <h1>Manatee LLM-Agent Trace GUI</h1>
              <p>새 workflow를 실행하고 SQLite/CSV/JSON trace를 한 화면에서 확인합니다.</p>
            </div>
            """
        )
        with gr.Tabs():
            with gr.Tab("실행"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=320):
                        source_stage = gr.Textbox(label="Source stage", value=CONFIG.source_stage)
                        target_stage = gr.Textbox(label="Target stage", value=CONFIG.target_stage)
                        models = gr.Textbox(label="Models", value=default_models, lines=4)
                        n_iter = gr.Slider(label="Iterations", minimum=1, maximum=10, step=1, value=CONFIG.n_iter)
                        n_candidates = gr.Slider(
                            label="Candidates per iteration",
                            minimum=2,
                            maximum=60,
                            step=1,
                            value=CONFIG.n_llm_candidates,
                        )
                        final_answer = gr.Checkbox(label="최종 한글 리포트 생성", value=False)
                        run_button = gr.Button("실행", variant="primary")
                    with gr.Column(scale=2):
                        query = gr.Textbox(label="Query", lines=5, value="")
                        status = gr.Textbox(label="Status", interactive=False)
                        run_dir_state = gr.State(value=None)
                        files = gr.Textbox(label="Saved files", lines=8, interactive=False)
                with gr.Row():
                    summary = gr.Dataframe(label="Summary CSV", interactive=False)
                trace = gr.JSON(label="Trace")

                run_button.click(
                    run_from_gui,
                    inputs=[query, source_stage, target_stage, models, n_iter, n_candidates, final_answer],
                    outputs=[status, summary, trace, files, run_dir_state],
                )

            with gr.Tab("이전 trace 조회"):
                with gr.Row():
                    run_select = gr.Dropdown(label="Run directory", choices=run_choices, value=run_choices[0] if run_choices else None)
                    refresh = gr.Button("목록 새로고침")
                    load = gr.Button("Trace 불러오기", variant="primary")
                previous_summary = gr.Dataframe(label="Summary CSV", interactive=False)
                previous_trace = gr.JSON(label="Trace")
                previous_files = gr.Textbox(label="Saved files", lines=10, interactive=False)

                refresh.click(refresh_runs, inputs=[], outputs=[run_select])
                load.click(load_run_trace, inputs=[run_select], outputs=[previous_summary, previous_trace, previous_files])

    return demo


def main() -> None:
    demo = build_demo()
    demo.queue()
    demo.launch()


if __name__ == "__main__":
    main()
