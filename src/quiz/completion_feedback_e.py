from openai import OpenAI
import os, json
from dotenv import load_dotenv


# === 환경 변수 ===
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# === 1. 의미 평가 (60%) ===
def evaluate_meaning(answer: str, userAnswer: str):
    prompt = f"""
당신은 뉴스 문장 평가 전문가이자 한국어 학습 피드백 코치입니다.
학습자의 문장이 모범답안의 의미를 얼마나 잘 전달했는지를 평가해주세요.

[평가 지침]
1. 핵심 의미가 완전히 반대일 경우 반드시 40점 이하로 평가하세요.
2. 의미와 방향성(긍정/부정/주장/반박 등)이 모두 일치하면 90점 이상.
3. 문장 표현이 다르더라도 의도나 논리 구조가 같다면 80점 이상을 부여하세요.
4. 주제는 같지만 강조점이나 방향이 약간 어긋나면 60~79점.
5. 핵심 개념이 다르거나 반대 논리이면 40~59점.
6. 거의 무관하거나 반대 의미일 경우 39점 이하.

[멘트]
1. 잘한 점과 개선할 점 순서대로 100자 이내로 피드백 해주세요.
2. 잘한 점은~ 개선할 점은~ 식의 말을 사용하지 마세요. 
3. 본론부터 이야기하세요.
4. 부드러운 문체의 존댓말 사용


모범답안: "{answer}"
학습자 문장: "{userAnswer}"

출력(JSON):
{{
  "score": (0~100 정수),
  "feedback": "100자 이내 피드백"
}}
"""
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(res.choices[0].message.content)
        return data["score"], data["feedback"]
    except Exception as e:
        print("의미 평가 오류:", e)
        return 0, "의미 평가 중 오류가 발생했습니다."


# === 2. 맥락 평가 (30%) ===
def evaluate_context(answer: str, userAnswer: str):
    prompt = f"""
당신은 뉴스 문맥 흐름 평가 전문가입니다.
학습자의 문장이 기사 전후 흐름과 얼마나 자연스럽게 이어지는지를 평가하세요.

[평가 지침]
1. 문장의 인과·설명·추론 관계가 모범답안과 같은 방향이라면 높은 점수.
2. 논리 전개가 어긋나거나, 인과가 반대되면 낮은 점수.
3. 문법적으로는 맞지만 의미상 기사 흐름에 맞지 않으면 60점 이하.
4. 기사 주제는 같지만 논리 전개가 어색하면 50~69점.
5. 의미·맥락이 완전히 반대거나 부자연스럽다면 40점 이하.
6. 모범답안의 전개 흐름을 자연스럽게 이어받고 있다면 90점 이상.

[멘트]
1. 잘한 점과 개선할 점 순서대로 100자 이내로 피드백 해주세요.
2. 잘한 점은~ 개선할 점은~ 식의 말을 사용하지 마세요. 
3. 본론부터 이야기하세요.
4. 부드러운 문체의 존댓말 사용

모범답안: "{answer}"
학습자 문장: "{userAnswer}"

출력(JSON):
{{
  "score": (0~100 정수),
  "feedback": "100자 이내 피드백"
}}
"""
    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(res.choices[0].message.content)
        return data["score"], data["feedback"]
    except Exception as e:
        print("맥락 평가 오류:", e)
        return 0, "맥락 평가 중 오류가 발생했습니다."


# === 3. 문법 평가 (10%) ===
def evaluate_grammar(userAnswer: str):
    prompt = f"""
너는 한국어 문장의 문법적 완성도를 평가하는 전문가야.
아래 문장의 문법적 정확성과 어법의 자연스러움을 0~100점으로 평가해.
피드백은 100자 이내로 작성해.

[평가 지침]
- 조사, 어미, 어순 오류가 많으면 낮은 점수 
- 명확하고 자연스러운 문장은 높은 점수

[멘트]
1. 잘한 점과 개선할 점 순서대로 100자 이내로 피드백 해주세요.
2. 잘한 점은~ 개선할 점은~ 식의 말을 사용하지 마세요. 
3. 본론부터 이야기하세요.
4. 부드러운 문체의 존댓말 사용

출력(JSON):
{{
  "score": (0~100 정수),
  "feedback": "100자 이내 피드백"
}}

[문장]
{userAnswer}
"""
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(res.choices[0].message.content)
        return data["score"], data["feedback"]
    except Exception as e:
        print("문법 평가 오류:", e)
        return 0, "문법 평가 중 오류가 발생했습니다."


# === 4. 총점 계산 및 출력 JSON 구성 ===
def evaluate_feedback(answer: str, userAnswer: str, question: str, topic: str, courseId: str, sessionId: str, level: str = "e"):
    meaning_score, meaning_fb = evaluate_meaning(answer, userAnswer)
    context_score, context_fb = evaluate_context(answer, userAnswer)
    grammar_score, grammar_fb = evaluate_grammar(userAnswer)

    # 가중치 적용
    total = int(0.6 * meaning_score + 0.3 * context_score + 0.1 * grammar_score)

    # 의미 점수 낮을 경우 cap
    if meaning_score < 50:
        total = min(total, 60)

    # 기본멘트 설정
    if total <= 40:
        default_feedback = "좀 더 생각해봐요"
    elif total <= 79:
        default_feedback = "좋아요, 이 부분만 추가하면 될 것 같아요"
    else:
        default_feedback = "오늘 열심히 학습했군요! 너무 잘했어요."

    # comment 통합 (맨 앞에 기본멘트 추가)
    comment = (
        f"{default_feedback}.\n"
        f"의미: {meaning_fb}\n"
        f"맥락: {context_fb}\n"
        f"문법: {grammar_fb}"
    )

    # 최종 출력 JSON 구조
    result = {
        "topic": topic,
        "courseId": courseId,
        "sessionId": sessionId,
        "contentType": "completionFeedback",
        "level": level,
        "items": [
            {
                "question": question,
                "answers": [
                    {
                        "userAnswer": userAnswer,
                        "score": total,
                        "comment": comment
                    }
                ]
            }
        ]
    }

    return result


# === 5. 실행 예시 ===
if __name__ == "__main__":
    topic = "politics"
    courseId = "1"
    sessionId = "11"
    level = "e"
    question = "인벤티지랩은 엠제이파트너스의 소송이 _______"
    answer = "전혀 근거 없는 것으로 판단하고 있다고 밝혔다."
    userAnswer = "근거 없는 주장이라고 밝혔다." 
    result = evaluate_feedback(answer, userAnswer, question, topic, courseId, sessionId, level)
    print(json.dumps(result, ensure_ascii=False, indent=2))