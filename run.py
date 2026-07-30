from __future__ import annotations

import argparse
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main.config import CONFIG, RunConfig


def default_query(source_stage: str, target_stage: str) -> str:
    return (
        f"{source_stage}에서 {target_stage} 상태로 전환시키기 위해, "
        "expression 변화와 이전 Manatee perturbation feedback을 참고해서 "
        "생물학적으로 타당하면서도 다양한 TF 후보와 조작 방향을 추천해줘."
    )


def parse_args() -> argparse.Namespace:
    def candidate_count(value: str) -> int:
        parsed = int(value)
        if parsed < 2 or parsed > 60:
            raise argparse.ArgumentTypeError("후보 수는 2 이상 60 이하로 지정하세요.")
        return parsed

    parser = argparse.ArgumentParser(description="Run the refactored Manatee LLM-Agent workflow.")
    parser.add_argument("--source-stage", default=CONFIG.source_stage)
    parser.add_argument("--target-stage", default=CONFIG.target_stage)
    parser.add_argument("--n-iter", type=int, default=CONFIG.n_iter)
    parser.add_argument("--n-candidates", type=candidate_count, default=CONFIG.n_llm_candidates)
    parser.add_argument("--model", action="append", dest="models", help="Model name. Repeat to run multiple models.")
    parser.add_argument("--query", default=None)
    parser.add_argument("--final-answer", action="store_true", help="Generate a Korean final report after screening.")
    parser.add_argument("--compute-mmd", action="store_true", help="Compute MMD metrics during every strategy evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from main.agent.runner import run_workflow

    config = RunConfig(
        source_stage=args.source_stage,
        target_stage=args.target_stage,
        n_iter=args.n_iter,
        n_llm_candidates=args.n_candidates,
        top_n_context=CONFIG.top_n_context,
        top_n_trrust=CONFIG.top_n_trrust,
        top_n_feedback_genes=CONFIG.top_n_feedback_genes,
        top_n_history=CONFIG.top_n_history,
        compute_mmd=args.compute_mmd or CONFIG.compute_mmd,
        use_all_tfs_as_allowed_list=CONFIG.use_all_tfs_as_allowed_list,
        models=tuple(args.models or CONFIG.models),
        exhaustive_csv=CONFIG.exhaustive_csv,
        results_root=CONFIG.results_root,
    )
    run_workflow(
        config=config,
        query=args.query or default_query(args.source_stage, args.target_stage),
        models=list(config.models),
        n_iter=args.n_iter,
        n_candidates=args.n_candidates,
        source_stage=args.source_stage,
        target_stage=args.target_stage,
        make_final_answer=args.final_answer,
    )


if __name__ == "__main__":
    main()
