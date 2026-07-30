from __future__ import annotations

from typing import Any


def format_feedback(history: list[dict[str, Any]], top_n_history: int) -> str:
    if not history:
        return "아직 이전 Manatee 피드백이 없습니다."

    lines = []
    for item in history[-top_n_history:]:
        lines.append(
            f"""
반복 {item["iteration"]} 피드백:
이전 최적 전략: {item["best_strategy"]}
거리 감소율: {item["distance_reduction_percent"]}
평균 cell 개선값: {item["mean_cell_improvement"]}
방향 cosine: {item["cosine_direction"]}

perturbation 이후 원본 source 대비 발현 변화:
상향 조절 상위 유전자:
{item["perturbed_deg"]["top_upregulated_genes"]}

하향 조절 상위 유전자:
{item["perturbed_deg"]["top_downregulated_genes"]}

해석 지침:
perturbation이 target stage 방향으로 이동했다면 성공한 TF 주변을 더 정교하게 탐색하세요.
실패했거나 반대 방향으로 이동했다면 약한 TF 조합 반복을 피하고 다른 TF를 탐색하세요.
"""
        )
    return "\n".join(lines)


def candidate_prompt(
    *,
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
) -> str:
    feedback_text = format_feedback(feedback_history, top_n_history)
    return f"""
당신은 JSON만 반환하는 API입니다.

반드시 유효한 JSON만 반환하세요.
마크다운, 설명문, 서문, 후문을 JSON 밖에 쓰지 마세요.
JSON 문자열 값 안에 줄바꿈 문자를 넣지 마세요.
"reason" 값은 한 줄의 짧은 문장이어야 합니다.

사용자 질문:
{query}

현재 반복:
{iteration} / {n_iter}

역할:
당신은 발생생물학과 single-cell transcriptomics 전문가입니다.

목표:
{source_stage} mouse embryo cell을 {target_stage} 상태에 가깝게 이동시킬 가능성이 있는 transcription factor perturbation 후보 {n_candidates}개를 생성하세요.

중요 규칙:
- 최종 2-TF pair를 직접 선택하지 마세요.
- 당신의 역할은 candidate perturbation pool만 생성하는 것입니다.
- 각 TF마다 선호 조작 방향을 하나만 제안하세요: "overexpress" 또는 "knockout".
- Manatee는 이후 당신이 제안한 방향과 반대 방향을 모두 평가합니다.
- exhaustive Manatee screen 결과는 제공되지 않습니다.
- exhaustive search ranking, random baseline 결과, global top perturbation pair를 사용하지 마세요.
- 생물학적 prior, 발현 변화, TRRUST 근거, 이전 반복의 Manatee 피드백만 사용하세요.
- TF 이름을 만들어내지 마세요.
- 반드시 허용 TF 목록에 있는 TF만 사용하세요.
- 같은 TF는 한 번만 포함하세요.
- candidate_perturbations 배열에는 정확히 {n_candidates}개 항목을 넣으세요.

선택 전략:
- canonical TF와 non-canonical TF를 모두 포함하세요.
- dataset-specific expression shift가 큰 TF를 포함하세요.
- 유명한 regulator만 고르지 마세요.
- 다음 근거 그룹에서 다양하게 고르세요:
  1. {target_stage}에서 강하게 상향 조절된 TF
  2. {target_stage}에서 강하게 하향 조절된 TF
  3. source와 target 사이 absolute expression change가 큰 TF
  4. TRRUST support가 있는 TF
  5. 이전 Manatee 피드백에서 탐색 가치가 생긴 TF
- 이전 perturbation이 전환을 개선했다면 성공 TF와 관련된 후보를 유지하되 새 대안도 추가하세요.
- 이전 perturbation이 약했다면 같은 약한 전략을 반복하지 마세요.

발현 기반 context:

{source_stage}에서 active한 상위 TF:
{context["top_active_source"]}

{target_stage}에서 active한 상위 TF:
{context["top_active_target"]}

{source_stage}에서 {target_stage}로 갈 때 증가한 상위 TF:
{context["top_upregulated_target"]}

{source_stage}에서 {target_stage}로 갈 때 감소한 상위 TF:
{context["top_downregulated_target"]}

DEG 유사 지표 기준 상위 TF 후보:
{context["top_deg_tfs"]}

TRRUST 근거가 있는 상위 TF:
{context["top_trrust_tfs"]}

이전 Manatee 피드백:
{feedback_text}

허용 TF 목록:
{allowed_tfs}

출력 JSON schema:
{{
  "candidate_perturbations": [
    {{"tf": "TF1", "action": "overexpress"}},
    {{"tf": "TF2", "action": "knockout"}}
  ],
  "reason": "후보 perturbation pool을 고른 짧은 이유"
}}

최종 지시:
JSON 객체 하나만 반환하세요.
첫 글자는 "{{" 이어야 합니다.
마지막 글자는 "}}" 이어야 합니다.
"""


def final_report_prompt(
    *,
    query: str,
    source_stage: str,
    target_stage: str,
    best_overall: dict[str, Any],
    history: list[dict[str, Any]],
) -> str:
    return f"""
당신은 Manatee single-cell perturbation 분석 도우미입니다.

반드시 한글로 답변하세요.

사용자 질문:
{query}

LLM-Agent + Manatee workflow가 완료되었습니다.

아래 구조로 명확한 과학 보고서 스타일의 요약을 작성하세요.

1. 추천 perturbation 전략
- 선택된 TF 조합을 설명하세요.
- knockout / overexpression 방향을 설명하세요.

2. Manatee 검증 결과
- perturbation 전 거리
- perturbation 후 거리
- 거리 감소율
- 평균 cell 개선값
- cosine 방향성
- perturbed cell이 target state 방향으로 이동했는지 해석하세요.

3. 반복 최적화 과정
- iteration 피드백이 어떻게 사용되었는지 간단히 설명하세요.

4. 생물학적 해석
- 선택된 TF의 가능한 생물학적 역할을 설명하세요.
- 데이터에서 확인된 근거와 일반적 생물학 해석을 구분하세요.
- 제공되지 않은 mechanism을 과장하지 마세요.

5. 최종 결론

분석 transition:
{source_stage} -> {target_stage}

최적 perturbation:
{best_overall}

반복 이력:
{history}

규칙:
- 제공된 수치만 사용하고 새 수치를 만들지 마세요.
- exhaustive rank가 제공되지 않았다면 언급하지 마세요.
- 간결하지만 정보가 충분한 답변을 작성하세요.
"""
