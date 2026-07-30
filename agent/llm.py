from __future__ import annotations

import json
import re
from typing import Any

from main.agent.prompts import candidate_prompt, final_report_prompt


VALID_ACTIONS = {"knockout", "overexpress"}


def _chat_ollama(*, model_name: str, num_ctx: int):
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise RuntimeError(
            "langchain_ollama 패키지를 찾지 못했습니다. 기존 base.py/api.py를 실행하던 "
            "Python 환경에서 실행하거나 해당 환경에 langchain-ollama를 설치하세요."
        ) from exc
    return ChatOllama(model=model_name, temperature=0, num_ctx=num_ctx, seed=42)


def extract_json(text: str) -> dict[str, Any]:
    raw = text.strip().replace("```json", "").replace("```", "")
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise ValueError(f"JSON을 추출하지 못했습니다. 원문:\n{text}")

    json_text = match.group(0)
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as exc:
        recovered = []
        arr_match = re.search(r'"candidate_perturbations"\s*:\s*\[(.*?)\]', json_text, re.S)
        if arr_match:
            for obj in re.findall(r"\{(.*?)\}", arr_match.group(1), re.S):
                tf_match = re.search(r'"tf"\s*:\s*"([^"]+)"', obj)
                action_match = re.search(r'"action"\s*:\s*"([^"]+)"', obj)
                if tf_match and action_match:
                    recovered.append({"tf": tf_match.group(1).strip(), "action": action_match.group(1).strip()})
        if not recovered:
            raise ValueError(f"JSON 파싱과 fallback 복구가 모두 실패했습니다. 원문:\n{text}") from exc
        return {
            "candidate_perturbations": recovered,
            "reason": "JSON parsing failed; candidates were recovered by regex.",
            "parse_warning": str(exc),
        }


def clean_candidate_perturbations(
    *,
    data: dict[str, Any],
    allowed_tfs: list[str],
    fallback_tfs: list[str],
    n_candidates: int,
) -> list[dict[str, str]]:
    allowed = set(allowed_tfs)
    cleaned = []
    seen_tfs = set()

    for item in data.get("candidate_perturbations", []):
        if not isinstance(item, dict):
            continue
        tf = str(item.get("tf", "")).strip()
        action = str(item.get("action", "")).strip()
        if tf in allowed and action in VALID_ACTIONS and tf not in seen_tfs:
            cleaned.append({"tf": tf, "action": action})
            seen_tfs.add(tf)

    for tf in fallback_tfs:
        if len(cleaned) >= n_candidates:
            break
        if tf in allowed and tf not in seen_tfs:
            cleaned.append({"tf": tf, "action": "overexpress"})
            seen_tfs.add(tf)

    if len(cleaned) < 2:
        raise ValueError(f"사용 가능한 perturbation 후보가 2개 미만입니다: {data}")

    return cleaned[:n_candidates]


def ask_llm_for_candidates(
    *,
    model_name: str,
    query: str,
    source_stage: str,
    target_stage: str,
    iteration: int,
    n_iter: int,
    n_candidates: int,
    context: dict[str, Any],
    allowed_tfs: list[str],
    feedback_history: list[dict[str, Any]],
    top_n_history: int,
) -> tuple[list[dict[str, str]], str, str]:
    prompt = candidate_prompt(
        query=query,
        source_stage=source_stage,
        target_stage=target_stage,
        iteration=iteration,
        n_iter=n_iter,
        n_candidates=n_candidates,
        context=context,
        allowed_tfs=allowed_tfs,
        feedback_history=feedback_history,
        top_n_history=top_n_history,
    )
    llm = _chat_ollama(model_name=model_name, num_ctx=8192)
    raw_response = llm.invoke(prompt).content
    data = extract_json(raw_response)
    candidates = clean_candidate_perturbations(
        data=data,
        allowed_tfs=allowed_tfs,
        fallback_tfs=context["fallback_tfs"],
        n_candidates=n_candidates,
    )
    return candidates, str(data.get("reason", "")), raw_response


def generate_final_answer(
    *,
    model_name: str,
    query: str,
    source_stage: str,
    target_stage: str,
    best_overall: dict[str, Any],
    history: list[dict[str, Any]],
) -> str:
    prompt = final_report_prompt(
        query=query,
        source_stage=source_stage,
        target_stage=target_stage,
        best_overall=best_overall,
        history=history,
    )
    llm = _chat_ollama(model_name=model_name, num_ctx=4096)
    return llm.invoke(prompt).content
