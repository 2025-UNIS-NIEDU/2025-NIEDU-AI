import os, json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# === 환경 변수 ===
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH, override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

QUIZ_DIR = BASE_DIR / "data" / "quiz"
today = datetime.now().strftime("%Y-%m-%d")

# === 메인 함수 ===
def generate_reflection_quiz(topic="politics"):
    """모든 단계 퀴즈(N/I/E)를 로드해 회고형(Reflection) 문제 자동 생성"""

    # === 파일 탐색 ===
    patterns = [
        f"{topic}_multi_ni_*.json",
        f"{topic}_short_ie_*.json",
        f"{topic}_completion_e_*.json",
    ]
    all_blocks = []
    for pattern in patterns:
        for file in QUIZ_DIR.glob(pattern):
            print(f"📂 로드 중: {file.name}")
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    all_blocks.extend(data)
            except json.JSONDecodeError:
                print(f"⚠️ JSON 오류: {file}")
                continue

    # === 데이터 병합 ===
    i_items, e_items = [], []
    course_id, session_id = None, None

    for block in all_blocks:
        if not isinstance(block, dict):
            continue
        if course_id is None:
            course_id = block.get("courseId")
        if session_id is None:
            session_id = block.get("sessionId")

        level = block.get("level")
        if level == "i":
            i_items.extend(block.get("items", []))
        elif level == "e":
            e_items.extend(block.get("items", []))

    print(f"\nI단계 {len(i_items)}개, E단계 {len(e_items)}개 로드 완료")

    # === 모델 ===
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # === 프롬프트 정의 ===
    def generate_quiz_i(i_quiz, summary=""):
        prompt_i = f"""
        당신은 뉴스 기반 학습 퀴즈 생성 AI입니다.
        아래는 동일 뉴스에 대해 이미 생성된 **I단계 문제** 목록입니다.
        이를 참고하여 학습자가 사건을 되돌아볼 수 있는 **회고형 질문 1개**를 만드세요.

        🎯 목표:
        - 기존 문제의 맥락을 반영하되, 뉴스의 본질적 의미를 되돌아보게 하는 사고형 질문
        - 정답이 정해져 있지 않은 개방형 질문
        - 단문 문어체 (20자 내외)
        - 반드시 JSON 형식을 따를 것

        ⚙️ 출력 예시(JSON):
        [
          {{
            "question": "이번 사안이 사회에 남긴 교훈은?"
          }}
        ]

        === I단계 문제 목록 ===
        {json.dumps(i_quiz, ensure_ascii=False, indent=2)}
        """
        res = llm.invoke(prompt_i)
        text = res.content.strip().replace("```json", "").replace("```", "")
        return json.loads(text)

    def generate_quiz_e(e_quiz, summary=""):
        prompt_e = f"""
        당신은 뉴스 기반 학습 설계 전문가입니다.
        아래는 동일 뉴스에 대해 이미 생성된 **E단계 문제** 목록입니다.
        이를 기반으로, 학습자가 사건의 원인·결과·의미를 성찰할 수 있는 **회고형 질문 1개**를 만드세요.

        🎯 목표:
        - 사회적 함의, 제도적 의미, 시사점 중심의 사고형 질문
        - 정답이 없는 개방형 문제
        - 문어체 20자 내외
        - 반드시 JSON 형식을 따를 것

        ⚙️ 출력 예시(JSON):
        [
          {{
            "question": "이 개편 논의가 제도 신뢰에 미친 영향은?"
          }}
        ]

        === E단계 문제 목록 ===
        {json.dumps(e_quiz, ensure_ascii=False, indent=2)}
        """
        res = llm.invoke(prompt_e)
        text = res.content.strip().replace("```json", "").replace("```", "")
        return json.loads(text)

    # === 실행 ===
    print("\nI단계 회고 문제 생성 중...")
    i_reflection = generate_quiz_i(i_items)
    print(json.dumps(i_reflection, ensure_ascii=False, indent=2))

    print("\nE단계 회고 문제 생성 중...")
    e_reflection = generate_quiz_e(e_items)
    print(json.dumps(e_reflection, ensure_ascii=False, indent=2))

    # === 저장 ===
    SAVE_DIR = QUIZ_DIR
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    final_result = [
        {
            "courseId": course_id,
            "sessionId": session_id,
            "topic": topic,
            "contentType": "reflection",
            "level": "i",
            "items": i_reflection,
        },
        {
            "courseId": course_id,
            "sessionId": session_id,
            "topic": topic,
            "contentType": "reflection",
            "level": "e",
            "items": e_reflection,
        },
    ]

    file_path = SAVE_DIR / f"{topic}_reflection_ie_{today}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    print(f"\nI/E 회고형 문제 저장 완료 → {file_path.resolve()}")
    return final_result