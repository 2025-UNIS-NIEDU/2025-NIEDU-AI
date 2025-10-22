import os, json, re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from select_session import select_session


# === 1. 환경 변수 설정 ===
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH, override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# === 2. 메인 함수 정의 ===
def generate_ox_quiz():
    """뉴스 요약문 기반 N단계 OX 퀴즈 5문항 생성"""
    
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
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # === 프롬프트 ===
    prompt_ox_n = f"""
    당신은 경제 뉴스 기반 학습용 퀴즈 생성 AI입니다.
    다음 뉴스 요약을 참고하여 **OX 퀴즈 5개 (전부 N단계)**를 만들어주세요.

    🎯 목표:
    - 입문자도 쉽게 풀 수 있는 **기초 사실 확인형(N단계)** 문제여야 합니다.
    - 문장은 간결하고 명확해야 하며, 기사 내용 그대로 또는 직관적으로 판단 가능한 수준이어야 합니다.

    규칙:
    1. 모든 문항은 명확하게 O 또는 X로 판단 가능한 문장이어야 합니다.
    2. 뉴스 summary의 사실만 사용하고 새로운 정보나 예측은 금지합니다.
    3. 문항들은 서로 다른 내용을 다뤄야 합니다.
    4. 문장은 짧고 명확하게, '~했다' / '~아니다' 형태로 작성하세요.
    5. 해설은 정답/오답 여부와 관계없이, 문항의 사실적 근거를 한 줄로 명확히 설명한다.
    6. 해설은 학습자에게 설명하듯 자연스럽게 작성, 50자 내외로.
    7. 출력은 아래 JSON 형식만으로 주세요:

    출력 형식(JSON):
    [
      {{
        "question": "문장 형태의 문제",
        "answer": "O 또는 X 중 하나",
        "isCorrect": "true" or "false",
        "explanation": "해설"
      }}
    ]

    문제 : 
    1. 단순 사실 확인 문제
    2. 기사와 반대 진술 문제
    3. 기사 속 조건과 반대 문제
    4. 문장 그대로 변형 문제
    5. 인과관계 단순화 반박 문제 

    뉴스 요약:
    {summary}
    """

    # === LLM 호출 ===
    response = llm.invoke(prompt_ox_n)
    text = response.content.strip().replace("```json", "").replace("```", "").strip()
    parsed_n = json.loads(text)

    # === NIEdu 구조 변환 ===
    ox_items = []
    for q in parsed_n:
        ans = q.get("answer", "").upper()
        option_O = {"text": "O", "isCorrect": ans == "O", "explanation": q.get("explanation", "")}
        option_X = {"text": "X", "isCorrect": ans == "X", "explanation": q.get("explanation", "")}
        ox_items.append({"question": q["question"], "answers": [option_O, option_X]})

    result = {
        "courseId": course_id,
        "sessionId": session_id,
        "topic": topic,
        "contentType": "ox",
        "level": "n",
        "items": ox_items
    }

    # === 결과 출력 ===
    print("\n=== 뉴스 요약문 ===\n")
    print(summary)
    print("\n=== 생성된 N단계 OX 퀴즈 ===\n")
    for i, item in enumerate(result["items"], 1):
        q = item["question"]
        correct_option = next((opt for opt in item["answers"] if opt["isCorrect"]), None)
        print(f"{i}. {q}")
        if correct_option:
            print(f"정답: {correct_option['text']} | {correct_option['explanation']}\n")
        else:
            print("정답 정보 없음\n")

    # === 저장 ===
    SAVE_DIR = BASE_DIR / "data" / "quiz"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    file_path = SAVE_DIR / f"{topic}_ox_n_{today}.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {file_path.resolve()}")
    return result