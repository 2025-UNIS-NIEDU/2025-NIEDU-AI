import os, json, random
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from select_session import select_session


# === 환경 변수 로드 ===
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH, override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# === 메인 함수 ===
def generate_completion_quiz():
    """뉴스 요약 기반 문장 완성형(E단계) 문제 자동 생성"""

    # === 세션 선택 ===
    selected_session = select_session()
    topic = selected_session["topic"]
    course_id = selected_session.get("courseId")
    session_id = selected_session.get("sessionId")
    summary = selected_session["summary"]

    print(f"\n선택된 토픽: {topic}")
    print(f"courseId: {course_id}")
    print(f"sessionId: {session_id}")
    print(f"제목: {selected_session.get('headline', '')}\n")

    # === 모델 설정 ===
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

    # === 퀴즈 생성 함수 ===
    def generate_sentence_completion_quiz(summary: str):
        prompt = f"""
        당신은 뉴스 기반 학습 퀴즈 생성 AI입니다.
        아래 뉴스 요약문을 바탕으로 **문장 완성형 단답식(E단계)** 문제 3개를 만들어주세요.

        🎯 목표:
        - 뉴스의 핵심 내용을 담은 문장을 완성하도록 유도
        - 학습자가 전체 문맥을 이해해야 자연스럽게 완성할 수 있어야 함
        - 정답은 완전한 문장(요약 기반)으로 작성
        - 해설은 생성하지 않음

        ⚙️ 규칙:
        - level="e"
        - question: 미완성 문장 형태 (약 30~40자, 마지막은 '...' 또는 불완전 문장으로 끝냄)
        - answer: 완성된 문장 (기사 요약의 사실 기반)
        - JSON 배열로 출력 (다른 텍스트 금지)
        - 기사 외 정보나 추측 금지

        출력 예시(JSON):
        [
          {{
            "question": "윤 의원은 수사권이 경찰로 집중되는 만큼 _______",
            "answers": [
              {{
                "text": "견제와 균형을 유지하는 제도적 장치가 필요하다고 말했다.",
                "isCorrect": true,
                "explanation": null
              }}
            ]
          }}
        ]

        뉴스 요약:
        {summary}

        이제 위 규칙에 따라 문장 완성형 문제 3개를 생성하세요.
        """
        res = llm.invoke(prompt)
        text = res.content.strip().replace("```json", "").replace("```", "")
        return json.loads(text)

    # === 실행 ===
    print("=== 뉴스 요약문 ===")
    print(summary)

    print("\n=== E단계 문제 생성 ===")
    e_quiz = generate_sentence_completion_quiz(summary)
    print(json.dumps(e_quiz, ensure_ascii=False, indent=2))

    print("\n [E단계 생성 결과]")
    print(json.dumps(e_quiz, ensure_ascii=False, indent=2))

    # === 저장 ===
    SAVE_DIR = BASE_DIR / "data" / "quiz"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")

    final_result = {
        "courseId": course_id,
        "sessionId": session_id,
        "topic": topic,
        "contentType": "completion",
        "level": "e",
        "items": e_quiz,
    }

    file_path = SAVE_DIR / f"{topic}_completion_e_{today}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    print(f"\n 전체 저장 완료 → {file_path.resolve()}")
    return final_result