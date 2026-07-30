from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from main.agent.llm import ask_llm_for_candidates, generate_final_answer
from main.agent.perturbation import (
    build_expression_context,
    compute_perturbed_deg,
    find_exhaustive_rank_for_strategy,
    save_csv,
    screen_candidate_pool,
)
from main.config import RunConfig
from main.fixed.data_adapter import ManateeData
from main.fixed.store import RunStore


def default_query(source_stage: str, target_stage: str) -> str:
    return (
        f"{source_stage}에서 {target_stage} 상태로 전환시키기 위해, "
        "expression 변화와 이전 Manatee perturbation feedback을 참고해서 "
        "생물학적으로 타당하면서도 다양한 TF 후보와 조작 방향을 추천해줘."
    )


def _safe_model_name(model_name: str) -> str:
    return model_name.replace(":", "_").replace("/", "_").replace("\\", "_")


def _json_dump(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
    return str(path)


def _summary_error_row(
    *,
    model_name: str,
    iteration: int,
    source_stage: str,
    target_stage: str,
    query: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "Model": model_name,
        "Iteration": iteration,
        "Transition": f"{source_stage}->{target_stage}",
        "User_Query": query,
        "LLM_Candidate_Perturbations": "ERROR",
        "LLM_Reason": str(error),
        "Candidate_Count": np.nan,
        "Local_Strategy_Count": np.nan,
        "Best_Local_Strategy": "ERROR",
        "Best_Local_Distance_Reduction": np.nan,
        "Best_Local_Mean_Cell_Improvement": np.nan,
        "Best_Local_MMD_Similarity": np.nan,
        "Best_Local_Target_Correlation": np.nan,
        "Best_Local_Cosine_Direction": np.nan,
        "Best_Matches_Both_LLM_Directions": np.nan,
        "Local_Screening_CSV": "ERROR",
        "Perturbed_DEG_CSV": "ERROR",
        "Exhaustive_Rank": np.nan,
        "Exhaustive_Percentile": np.nan,
        "Exhaustive_Distance_Reduction": np.nan,
        "Exhaustive_Matched_Strategy": "ERROR",
    }


def run_workflow(
    *,
    config: RunConfig,
    query: str | None = None,
    models: list[str] | None = None,
    n_iter: int | None = None,
    n_candidates: int | None = None,
    source_stage: str | None = None,
    target_stage: str | None = None,
    make_final_answer: bool = False,
) -> dict[str, Any]:
    source_stage = source_stage or config.source_stage
    target_stage = target_stage or config.target_stage
    n_iter = n_iter or config.n_iter
    n_candidates = n_candidates or config.n_llm_candidates
    models = models or list(config.models)
    query = query or default_query(source_stage, target_stage)

    run_dir = config.make_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    store = RunStore(run_dir / "manatee_run.sqlite3")
    run_id = store.create_run(source_stage, target_stage, models, n_iter, n_candidates, run_dir)

    data = ManateeData.from_api_module()
    source_idx, target_idx = data.stage_indices(source_stage, target_stage)
    context, context_df = build_expression_context(data, source_stage, target_stage, config)
    context_csv = save_csv(context_df, run_dir / "expression_context_tf_table.csv")
    mu_all_np = data.latent()
    valid_tfs = data.valid_tfs()

    raw_outputs: dict[str, list[dict[str, Any]]] = {}
    full_history: dict[str, list[dict[str, Any]]] = {}
    summary_rows: list[dict[str, Any]] = []

    print("===== Iterative LLM-Agent + Manatee Validation =====")
    print(f"Transition: {source_stage} -> {target_stage}")
    print("주의: exhaustive rank/top 조합은 LLM 입력에 사용하지 않습니다.")
    print("LLM은 TF 후보와 권장 방향만 제안하고, Manatee가 양방향 perturbation을 평가합니다.")
    print(f"Source cells ({source_stage}): {len(source_idx)}")
    print(f"Target cells ({target_stage}): {len(target_idx)}")
    print(f"Valid TFs: {len(valid_tfs)}")
    print(f"Output dir: {run_dir}")

    try:
        for model_name in models:
            print(f"\n==============================\nMODEL: {model_name}\n==============================")
            model_history: list[dict[str, Any]] = []
            raw_outputs[model_name] = []

            for iteration in range(1, n_iter + 1):
                print(f"\n----- Iteration {iteration}/{n_iter} -----")
                iteration_id = None
                try:
                    candidates, reason, raw_response = ask_llm_for_candidates(
                        model_name=model_name,
                        query=query,
                        source_stage=source_stage,
                        target_stage=target_stage,
                        iteration=iteration,
                        n_iter=n_iter,
                        n_candidates=n_candidates,
                        context=context,
                        allowed_tfs=valid_tfs,
                        feedback_history=model_history,
                        top_n_history=config.top_n_history,
                    )
                    raw_outputs[model_name].append({"iteration": iteration, "raw_response": raw_response})

                    screen_df = screen_candidate_pool(
                        data=data,
                        candidate_perturbations=candidates,
                        mu_all_np=mu_all_np,
                        source_stage=source_stage,
                        target_stage=target_stage,
                        compute_mmd=config.compute_mmd,
                    )
                    safe_model = _safe_model_name(model_name)
                    screen_csv = save_csv(screen_df, run_dir / f"{safe_model}_iter{iteration}_screening_with_opposite.csv")

                    best = screen_df.iloc[0].to_dict()
                    rank_info = find_exhaustive_rank_for_strategy(best, config.exhaustive_csv)
                    deg_feedback, deg_df = compute_perturbed_deg(
                        data=data,
                        best_row=best,
                        source_stage=source_stage,
                        top_n_feedback_genes=config.top_n_feedback_genes,
                    )
                    deg_csv = save_csv(deg_df, run_dir / f"{safe_model}_iter{iteration}_perturbed_vs_original_deg.csv")

                    history_item = {
                        "model": model_name,
                        "iteration": iteration,
                        "candidate_perturbations": candidates,
                        "llm_reason": reason,
                        "best_strategy": best["strategy"],
                        "tf1": best["tf1"],
                        "action1": best["action1"],
                        "tf2": best["tf2"],
                        "action2": best["action2"],
                        "distance_reduction_percent": best["distance_reduction_percent"],
                        "mean_cell_improvement": best["mean_cell_improvement"],
                        "mmd_similarity": best.get("mmd_similarity"),
                        "target_correlation": best["target_correlation"],
                        "cosine_direction": best["cosine_direction"],
                        "matches_both_llm_directions": best["matches_both_llm_directions"],
                        "perturbed_deg": deg_feedback,
                        "screening_csv": screen_csv,
                        "deg_csv": deg_csv,
                        "rank_info": rank_info,
                    }
                    model_history.append(history_item)

                    metrics = {
                        "distance_reduction_percent": best["distance_reduction_percent"],
                        "mean_cell_improvement": best["mean_cell_improvement"],
                        "mmd_similarity": best.get("mmd_similarity"),
                        "target_correlation": best["target_correlation"],
                        "cosine_direction": best["cosine_direction"],
                        "matches_both_llm_directions": best["matches_both_llm_directions"],
                        **rank_info,
                    }
                    iteration_id = store.add_iteration(
                        run_id=run_id,
                        model_name=model_name,
                        iteration=iteration,
                        llm_reason=reason,
                        best_strategy=best["strategy"],
                        metrics=metrics,
                        screening_csv=screen_csv,
                        deg_csv=deg_csv,
                    )
                    store.add_candidates(iteration_id, candidates)
                    store.add_raw_output(iteration_id, raw_response)
                    store.add_dataframe_rows("screening_results", iteration_id, screen_df, limit=200)
                    store.add_dataframe_rows("deg_results", iteration_id, deg_df, limit=500)

                    candidate_text = ";".join([f'{item["tf"]}({item["action"]})' for item in candidates])
                    summary_rows.append(
                        {
                            "Model": model_name,
                            "Iteration": iteration,
                            "Transition": f"{source_stage}->{target_stage}",
                            "User_Query": query,
                            "LLM_Candidate_Perturbations": candidate_text,
                            "LLM_Reason": reason,
                            "Candidate_Count": len(candidates),
                            "Local_Strategy_Count": len(screen_df),
                            "Best_Local_Strategy": best["strategy"],
                            "Best_Local_Distance_Reduction": best["distance_reduction_percent"],
                            "Best_Local_Mean_Cell_Improvement": best["mean_cell_improvement"],
                            "Best_Local_MMD_Similarity": best.get("mmd_similarity"),
                            "Best_Local_Target_Correlation": best["target_correlation"],
                            "Best_Local_Cosine_Direction": best["cosine_direction"],
                            "Best_Matches_Both_LLM_Directions": best["matches_both_llm_directions"],
                            "Local_Screening_CSV": screen_csv,
                            "Perturbed_DEG_CSV": deg_csv,
                            **rank_info,
                        }
                    )
                    print(f"Best Manatee-validated strategy: {best['strategy']}")
                    print(f"distance_reduction_percent: {best['distance_reduction_percent']}")
                    print(f"rank_info: {rank_info}")

                except Exception as exc:
                    print(f"ERROR: {exc}")
                    if iteration_id is None:
                        store.add_iteration(
                            run_id=run_id,
                            model_name=model_name,
                            iteration=iteration,
                            error=str(exc),
                        )
                    summary_rows.append(
                        _summary_error_row(
                            model_name=model_name,
                            iteration=iteration,
                            source_stage=source_stage,
                            target_stage=target_stage,
                            query=query,
                            error=exc,
                        )
                    )

            full_history[model_name] = model_history

        summary_df = pd.DataFrame(summary_rows)
        summary_csv = save_csv(summary_df, run_dir / "iterative_summary.csv")
        raw_json = _json_dump(run_dir / "raw_llm_outputs.json", raw_outputs)
        history_json = _json_dump(run_dir / "iteration_history.json", full_history)

        final_answer = None
        successful = summary_df[summary_df["Best_Local_Strategy"] != "ERROR"] if not summary_df.empty else pd.DataFrame()
        if make_final_answer and not successful.empty:
            best_overall = successful.sort_values(
                ["Best_Local_Distance_Reduction", "Best_Local_Mean_Cell_Improvement"],
                ascending=False,
            ).iloc[0].to_dict()
            final_answer = generate_final_answer(
                model_name=models[0],
                query=query,
                source_stage=source_stage,
                target_stage=target_stage,
                best_overall=best_overall,
                history=full_history,
            )
            _json_dump(run_dir / "final_answer.json", {"answer": final_answer, "best_overall": best_overall})

        store.finish_run(run_id, "completed")
        print("\n===== 최종 요약 =====")
        print(summary_df.to_string(index=False))
        print("\n저장 완료:")
        print(summary_csv)
        print(raw_json)
        print(history_json)

        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "sqlite": str(run_dir / "manatee_run.sqlite3"),
            "context_csv": context_csv,
            "summary_csv": summary_csv,
            "raw_json": raw_json,
            "history_json": history_json,
            "final_answer": final_answer,
        }
    except Exception:
        store.finish_run(run_id, "failed")
        raise
