from openai import OpenAI
import os, json
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

def generate_completion_feedback_quiz():
    # === 환경 변수 ===
    load_dotenv(override=True)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)

    # === 1. 의미 평가 (60%) ===
    def evaluate_meaning(answer: str, userAnswer: str):
        prompt = f"""
    당신은 뉴스 문장 평가 전문가이자 한국어 학습 피드백 코치입니다.
    학습자의 문장이 모범답안의 의미를 얼마나 잘 전달했는지를 평가해주세요.

    [평가 절차]  
    1. 모범답안의 핵심 의미(주장, 결론, 방향성)를 간단히 파악합니다.  
    2. 학습자의 문장이 이 의미를 올바르게 전달했는지만 평가합니다.  
    3. 문체, 어법, 세부 표현의 차이는 감점하지 않습니다.  
    4. 의미가 동일하거나 거의 동일하다면 90점 이상을 부여합니다.  
    5. 완전히 반대되거나 왜곡된 경우에만 큰 감점을 적용합니다.  

    [평가 기준]  
    1. 의미가 동일하거나 거의 동일 (핵심 주장 또는 방향성 일치): **90~100점**  
    2. 핵심 방향은 같으나 표현이 간략하거나 일부 누락된 경우: **80~89점**  
    3. 주제는 같으나 논리 방향이 약간 어긋남: **60~79점**  
    4. 핵심 의미가 다르거나 반대 의미로 표현됨: **0~59점**

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
                temperature=0.0,
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

    [평가 절차]
    1. 모범답안이 어떤 맥락(인과·설명·추론·전환)을 형성하는 문장인지 요약합니다.  
    예: 원인 제시 / 결과 설명 / 반박 / 보충 / 결론 / 전환 등
    2. 학습자의 문장이 그 흐름을 얼마나 자연스럽게 이어가는지 판단합니다.
    3. 문법적으로 맞더라도 논리나 흐름이 어색하면 감점합니다.

    [판단 기준]
    - 인과 관계: 사건의 원인→결과 순서가 자연스럽게 이어지는가?
    - 설명 관계: 앞 문장의 주제나 근거를 부연·보완하고 있는가?
    - 추론 관계: 이전 정보로부터 합리적으로 도출되는 내용인가?
    - 전환 관계: 다른 시각이나 새로운 정보로 자연스럽게 넘어가는가?

    [평가 기준 1 : 정답에 키워드 없음]
    - 전후 문맥과 인과·추론·설명 흐름이 완벽히 일치: 90~100점  
    - 대체로 자연스럽지만 세부 논리가 다소 생략됨: 80~89점  
    - 주제는 같지만 논리 전개가 어색하거나 불연속적: 60~79점  
    - 문법은 맞지만 의미상 기사 흐름에 맞지 않음: 40~59점  
    - 인과나 논리가 반대·역행하거나 완전히 단절됨: 0~39점  

    [평가기준 2 — 키워드 포함형]
    - 핵심 키워드(정답의 주요 명사·개념어)가 모두 포함되고 의미도 동일함 → 90~100점
    - 핵심 키워드 일부 누락 또는 유사어로 대체되었으나 의미 유지 → 80~89점
    - 주제는 같지만 핵심 키워드 대부분 누락 → 60~79점
    - 핵심 키워드 누락으로 의미 왜곡 또는 모순 발생 → 0~59점 

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
                temperature=0.0,
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
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            data = json.loads(res.choices[0].message.content)
            return data["score"], data["feedback"]
        except Exception as e:
            print("문법 평가 오류:", e)
            return 0, "문법 평가 중 오류가 발생했습니다."


    # === 4. 총점 계산 및 출력 JSON 구성 ===
    def evaluate_feedback(answer: str, userAnswer: str, question: str, contentId: int, level: str = "E"):
        meaning_score, meaning_fb = evaluate_meaning(answer, userAnswer)
        context_score, context_fb = evaluate_context(answer, userAnswer)
        grammar_score, grammar_fb = evaluate_grammar(userAnswer)

        # 가중치 적용
        total = int(0.6 * meaning_score + 0.3 * context_score + 0.1 * grammar_score)

        # 점수 십의 자리 단위
        total = int(round(total / 10) * 10)

        # comment 통합 (맨 앞에 기본멘트 추가)
        comment = (
            f"의미: {meaning_fb}\n"
            f"맥락: {context_fb}\n"
            f"문법: {grammar_fb}"
        )

        # 🔹 리스트 없이 단일 객체 반환
        return {
            "contentId": contentId,
            "question": question,
            "userAnswer": userAnswer,
            "score": total,
            "comment": comment
        }
    

    # === 여러 문항 자동 평가 ===
    if __name__ == "__main__":
        topic = "politics"
        courseId = "1"
        sessionId = "1"
        level = "E"

        qa_list = [
            {
                "contentId": 1,
                "question": "한국과 싱가포르가 ______",
                "referenceAnswer": "전략적 동반자 관계를 수립했습니다.",
                "userAnswer": "긴밀한 관계를 끊었습니다."  
            },
            {
                "contentId": 2,
                "question": "이재명 대통령은 정상회담 결과를 ______",
                "referenceAnswer": "공동언론발표에서 설명했습니다.",
                "userAnswer": "언론에서 발표했습니다."  
            },
            {
                "contentId": 3,
                "question": "웡 총리는 양국 관계의 훌륭한 상태를 ______",
                "referenceAnswer": "점검하고 앞으로 더 나은 관계를 맺을 수 있음을 확인했습니다.",
                "userAnswer": "점검하고 앞으로 더 나은 관계를 맺을 수 있음을 확인했습니다." 
            }
        ]

        results = [
            evaluate_feedback(q["referenceAnswer"], q["userAnswer"], q["question"], q["contentId"], level)
            for q in qa_list
        ]

        final_output = {
            "contentType": "COMPLETION_FEEDBACK",
            "level": level,
            "contents": results
        }

        print(json.dumps(final_output, ensure_ascii=False, indent=2))

        # === 저장 ===
        BASE_DIR = Path(__file__).resolve().parents[2]
        SAVE_DIR = BASE_DIR / "data" / "quiz"
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

        today = datetime.now().strftime("%Y-%m-%d")
        file_name = f"{topic}_{courseId}_{sessionId}_FEEDBACK_{today}.json"
        save_path = SAVE_DIR / file_name

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)

        print(f"결과 저장 완료: {save_path}")

#  실행
if __name__ == "__main__":
    generate_completion_feedback_quiz()