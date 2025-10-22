from openai import OpenAI
import os, json
from dotenv import load_dotenv

# === 환경 변수 로드 ===
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. 문법 평가
def evaluate_grammar(user_answer: str) -> int:
    prompt = f"""
너는 한국어 문장의 문법적 완성도를 평가하는 전문가야.
아래 문장의 문법적 정확성과 어법의 자연스러움을 0~100점으로 평가해.

평가 기준:
- 조사, 어미, 어순 오류가 많으면 낮은 점수
- 명확하고 자연스러운 문장은 높은 점수
- 출력은 반드시 JSON 형식으로 작성: {{"score": int}}

[문장]
{user_answer}
"""
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        score = int(json.loads(res.choices[0].message.content)["score"])
    except Exception:
        score = 0
    return score


# 2. 맥락 평가
def evaluate_context(user_answer: str, reference_text: str) -> int:
    prompt = f"""
너는 의미 일치도를 평가하는 전문가야.
다음 두 문장의 핵심 의미가 얼마나 유사한지 0~100점으로 평가하라.
단순한 단어 일치가 아니라, 말하고자 하는 핵심 뜻이 같은지를 기준으로 점수를 매겨라.

평가 기준:
- 의미가 거의 다르면 40점 이하
- 일부만 맞으면 60~79점
- 핵심 의미가 같으면 80점 이상
- 출력은 반드시 JSON 형식으로 작성: {{"score": int}}

[문맥(reference)]
{reference_text}

[사용자 문장(user)]
{user_answer}
"""
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        score = int(json.loads(res.choices[0].message.content)["score"])
    except Exception:
        score = 0
    return score


# 3. 총점 계산
def evaluate_total(user_answer: str, reference_text: str) -> dict:
    """
    문법 + 맥락 평가를 종합하여 총점 계산
    - 문법: 30%
    - 맥락: 70%
    """
    grammar = evaluate_grammar(user_answer)
    context = evaluate_context(user_answer, reference_text)
    total = int(grammar * 0.3 + context * 0.7)

    feedback = {
        "grammarScore": grammar,
        "contextScore": context,
        "totalScore": total,
    }
    return feedback

# 4. 피드백 생성 (점수 기반 + AI 보정)
def generate_feedback(user_answer: str, reference_text: str) -> dict:
    """
    최종 점수를 기반으로 한 피드백 생성
    - 0~40: "좀 더 생각해봐요"
    - 41~79: "좋아요, 이 부분만 추가하면 될 것 같아요"
    - 80~100: "오늘 열심히 학습했군요! 너무 잘했어요."
    - 피드백 문구는 LLM이 100자 이내로 자연스럽게 보완
    """

    scores = evaluate_total(user_answer, reference_text)
    total = scores["totalScore"]

    if total <= 40:
        base_comment = "좀 더 생각해봐요. 핵심 의미를 놓친 부분이 있어요."
    elif total <= 79:
        base_comment = "좋아요, 이 부분만 추가하면 될 것 같아요."
    else:
        base_comment = "오늘 열심히 학습했군요! 너무 잘했어요."

    # LLM으로 자연스러운 문장 보정 (100자 이내)
    refine_prompt = f"""
다음 문장을 학습 피드백으로 자연스럽게 다듬어줘.
최대 100자 이내로, 따뜻하고 구체적인 문체로 써줘.

[기본 피드백]
{base_comment}

[문장 비교]
기준 문장: {reference_text}
사용자 문장: {user_answer}
"""
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": refine_prompt}],
        )
        refined = res.choices[0].message.content.strip()
    except Exception:
        refined = base_comment

    return {
        "userAnswer": user_answer,
        "score": total,
        "comment": refined
    }