from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

from main.config import RunConfig
from main.fixed.data_adapter import ManateeData


def opposite_action(action: str) -> str:
    if action == "overexpress":
        return "knockout"
    if action == "knockout":
        return "overexpress"
    raise ValueError(f"알 수 없는 action입니다: {action}")


def build_expression_context(
    data: ManateeData,
    source_stage: str,
    target_stage: str,
    config: RunConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    source_idx, target_idx = data.stage_indices(source_stage, target_stage)
    gene_idx = data.gene_name_to_idx()
    source_mean = data.x[source_idx].mean(axis=0)
    target_mean = data.x[target_idx].mean(axis=0)

    rows = []
    for tf in data.tfs:
        if tf not in gene_idx:
            continue
        idx = gene_idx[tf]
        source_expr = float(source_mean[idx])
        target_expr = float(target_mean[idx])
        delta = target_expr - source_expr
        rows.append(
            {
                "tf": tf,
                "source_expr": source_expr,
                "target_expr": target_expr,
                "delta_expr": delta,
                "abs_delta_expr": abs(delta),
                "trrust_targets": int(len(data.trrust[tf])) if tf in data.trrust else 0,
            }
        )

    df = pd.DataFrame(rows)
    df["priority_score"] = df["abs_delta_expr"].rank(pct=True) + df["trrust_targets"].rank(pct=True)

    def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        return frame.round(6).to_dict("records")

    context = {
        "top_active_source": records(df.sort_values("source_expr", ascending=False).head(config.top_n_context)),
        "top_active_target": records(df.sort_values("target_expr", ascending=False).head(config.top_n_context)),
        "top_upregulated_target": records(df.sort_values("delta_expr", ascending=False).head(config.top_n_context)),
        "top_downregulated_target": records(df.sort_values("delta_expr", ascending=True).head(config.top_n_context)),
        "top_deg_tfs": records(df.sort_values("abs_delta_expr", ascending=False).head(config.top_n_context)),
        "top_trrust_tfs": records(df.sort_values("trrust_targets", ascending=False).head(config.top_n_trrust)),
        "fallback_tfs": df.sort_values("priority_score", ascending=False)["tf"].head(config.n_llm_candidates).tolist(),
    }
    return context, df


def apply_perturbation(
    x_source: Any,
    perturbations: list[tuple[str, str]],
    gene_name_to_idx: dict[str, int],
) -> Any | None:
    perturbed = x_source.copy()
    for tf, action in perturbations:
        idx = gene_name_to_idx.get(tf)
        if idx is None:
            return None
        if action == "knockout":
            perturbed[:, idx] = 0.0
        elif action == "overexpress":
            perturbed[:, idx] = np.where(perturbed[:, idx] > 0, perturbed[:, idx] * 3.0, 2.5)
        else:
            return None
    return perturbed


def evaluate_two_tf_strategy(
    *,
    data: ManateeData,
    tf1: str,
    action1: str,
    tf2: str,
    action2: str,
    source_stage: str,
    target_stage: str,
    mu_all_np: np.ndarray,
    compute_mmd: bool,
) -> dict[str, Any] | None:
    source_idx, target_idx = data.stage_indices(source_stage, target_stage)
    gene_idx = data.gene_name_to_idx()
    x_source = data.x[source_idx].copy()
    pert_x = apply_perturbation(x_source, [(tf1, action1), (tf2, action2)], gene_idx)
    if pert_x is None:
        return None

    original_z = mu_all_np[source_idx]
    target_z = mu_all_np[target_idx]
    orig_c = original_z.mean(axis=0)
    targ_c = target_z.mean(axis=0)
    dist_before = float(np.linalg.norm(orig_c - targ_c))

    perturbed_z = data.latent(pert_x)
    pert_c = perturbed_z.mean(axis=0)
    dist_after = float(np.linalg.norm(pert_c - targ_c))
    dist_reduction = ((dist_before - dist_after) / dist_before) * 100 if dist_before > 0 else 0.0

    cell_before = np.linalg.norm(original_z - targ_c, axis=1)
    cell_after = np.linalg.norm(perturbed_z - targ_c, axis=1)
    mean_cell_improvement = float(np.mean(cell_before - cell_after))

    ideal_v = targ_c - orig_c
    actual_v = pert_c - orig_c
    if np.linalg.norm(ideal_v) > 0 and np.linalg.norm(actual_v) > 0:
        cosine_direction = float(1 - cosine(ideal_v, actual_v))
    else:
        cosine_direction = 0.0

    if compute_mmd:
        target_z_sub = target_z[: len(perturbed_z)]
        xx = np.exp(-np.sum((perturbed_z[:, None, :] - perturbed_z[None, :, :]) ** 2, axis=2)).mean()
        yy = np.exp(-np.sum((target_z_sub[:, None, :] - target_z_sub[None, :, :]) ** 2, axis=2)).mean()
        xy = np.exp(-np.sum((perturbed_z[:, None, :] - target_z_sub[None, :, :]) ** 2, axis=2)).mean()
        mmd = float(max(xx + yy - 2 * xy, 0.0))
        mmd_similarity = float(1 / (1 + mmd))
    else:
        mmd = np.nan
        mmd_similarity = np.nan

    corr, _ = pearsonr(pert_c, targ_c)
    if np.isnan(corr):
        corr = 0.0

    return {
        "tf1": tf1,
        "action1": action1,
        "tf2": tf2,
        "action2": action2,
        "strategy": f"{tf1}({action1}) + {tf2}({action2})",
        "source_stage": source_stage,
        "target_stage": target_stage,
        "distance_before": round(dist_before, 6),
        "distance_after": round(dist_after, 6),
        "distance_reduction_percent": round(float(dist_reduction), 6),
        "mean_cell_improvement": round(mean_cell_improvement, 6),
        "mmd": round(mmd, 6),
        "mmd_similarity": round(mmd_similarity, 6),
        "target_correlation": round(float(corr), 6),
        "cosine_direction": round(cosine_direction, 6),
    }


def screen_candidate_pool(
    *,
    data: ManateeData,
    candidate_perturbations: list[dict[str, str]],
    mu_all_np: np.ndarray,
    source_stage: str,
    target_stage: str,
    compute_mmd: bool,
) -> pd.DataFrame:
    gene_idx = data.gene_name_to_idx()
    valid_candidates = [
        item
        for item in candidate_perturbations
        if item["tf"] in gene_idx and item["action"] in {"knockout", "overexpress"}
    ]
    tf_to_action = {item["tf"]: item["action"] for item in valid_candidates}
    results = []

    for tf1, tf2 in itertools.combinations(tf_to_action.keys(), 2):
        suggested_a1 = tf_to_action[tf1]
        suggested_a2 = tf_to_action[tf2]
        for action1 in [suggested_a1, opposite_action(suggested_a1)]:
            for action2 in [suggested_a2, opposite_action(suggested_a2)]:
                result = evaluate_two_tf_strategy(
                    data=data,
                    tf1=tf1,
                    action1=action1,
                    tf2=tf2,
                    action2=action2,
                    source_stage=source_stage,
                    target_stage=target_stage,
                    mu_all_np=mu_all_np,
                    compute_mmd=compute_mmd,
                )
                if result is None:
                    continue
                result["llm_suggested_action1"] = suggested_a1
                result["llm_suggested_action2"] = suggested_a2
                result["matches_llm_direction_tf1"] = action1 == suggested_a1
                result["matches_llm_direction_tf2"] = action2 == suggested_a2
                result["matches_both_llm_directions"] = action1 == suggested_a1 and action2 == suggested_a2
                results.append(result)

    df = pd.DataFrame(results)
    if df.empty:
        raise ValueError("screening 결과가 비어 있습니다.")
    return df.sort_values(
        by=["distance_reduction_percent", "mean_cell_improvement"],
        ascending=False,
    ).reset_index(drop=True)


def compute_perturbed_deg(
    *,
    data: ManateeData,
    best_row: dict[str, Any],
    source_stage: str,
    top_n_feedback_genes: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    source_idx = np.where(data.labels == source_stage)[0]
    gene_idx = data.gene_name_to_idx()
    x_source = data.x[source_idx].copy()
    pert_x = apply_perturbation(
        x_source,
        [(best_row["tf1"], best_row["action1"]), (best_row["tf2"], best_row["action2"])],
        gene_idx,
    )
    if pert_x is None:
        raise ValueError("perturbed expression을 생성하지 못했습니다.")

    original_mean = x_source.mean(axis=0)
    perturbed_mean = pert_x.mean(axis=0)
    delta = perturbed_mean - original_mean
    abs_delta = np.abs(delta)

    rows = []
    tf_set = set(data.tfs)
    for gene, idx in gene_idx.items():
        rows.append(
            {
                "gene": gene,
                "original_expr": float(original_mean[idx]),
                "perturbed_expr": float(perturbed_mean[idx]),
                "delta_perturbed_vs_original": float(delta[idx]),
                "abs_delta": float(abs_delta[idx]),
                "is_tf": gene in tf_set,
            }
        )
    df = pd.DataFrame(rows)

    def compact(frame: pd.DataFrame) -> list[dict[str, Any]]:
        return [
            {
                "gene": row["gene"],
                "delta": round(float(row["delta_perturbed_vs_original"]), 4),
                "original": round(float(row["original_expr"]), 4),
                "perturbed": round(float(row["perturbed_expr"]), 4),
            }
            for _, row in frame.iterrows()
        ]

    deg_feedback = {
        "top_upregulated_genes": compact(
            df.sort_values("delta_perturbed_vs_original", ascending=False).head(top_n_feedback_genes)
        ),
        "top_downregulated_genes": compact(
            df.sort_values("delta_perturbed_vs_original", ascending=True).head(top_n_feedback_genes)
        ),
        "top_upregulated_tfs": compact(
            df[df["is_tf"]].sort_values("delta_perturbed_vs_original", ascending=False).head(top_n_feedback_genes)
        ),
        "top_downregulated_tfs": compact(
            df[df["is_tf"]].sort_values("delta_perturbed_vs_original", ascending=True).head(top_n_feedback_genes)
        ),
    }
    return deg_feedback, df


def find_exhaustive_rank_for_strategy(best_row: dict[str, Any], exhaustive_csv: Path) -> dict[str, Any]:
    if not exhaustive_csv.exists():
        return {
            "Exhaustive_Rank": np.nan,
            "Exhaustive_Percentile": np.nan,
            "Exhaustive_Distance_Reduction": np.nan,
            "Exhaustive_Matched_Strategy": "NOT_AVAILABLE",
        }

    df = pd.read_csv(exhaustive_csv)
    df = df.sort_values(
        by=["distance_reduction_percent", "mean_cell_improvement"],
        ascending=False,
    ).reset_index(drop=True)

    tf1, tf2 = best_row["tf1"], best_row["tf2"]
    a1, a2 = best_row["action1"], best_row["action2"]
    direct = (df["tf1"] == tf1) & (df["action1"] == a1) & (df["tf2"] == tf2) & (df["action2"] == a2)
    reverse = (df["tf1"] == tf2) & (df["action1"] == a2) & (df["tf2"] == tf1) & (df["action2"] == a1)
    matched = df[direct | reverse]
    if matched.empty:
        return {
            "Exhaustive_Rank": np.nan,
            "Exhaustive_Percentile": np.nan,
            "Exhaustive_Distance_Reduction": np.nan,
            "Exhaustive_Matched_Strategy": "NOT_FOUND",
        }

    idx = int(matched.index[0])
    rank = idx + 1
    return {
        "Exhaustive_Rank": rank,
        "Exhaustive_Percentile": round(float(rank / len(df) * 100), 6),
        "Exhaustive_Distance_Reduction": round(float(matched.iloc[0]["distance_reduction_percent"]), 6),
        "Exhaustive_Matched_Strategy": matched.iloc[0].get("strategy", ""),
    }


def save_csv(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)
