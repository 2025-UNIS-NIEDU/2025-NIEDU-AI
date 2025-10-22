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
def generate_short_quiz():
    """뉴스 요약 기반 I + E 단답형 문제 자동 생성"""

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
    llm_i = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    llm_e = ChatOpenAI(model="gpt-5")

    # === I단계 문제 생성 ===
    def generate_quiz_i(summary: str):
        prompt_i = f"""
        당신은 뉴스 기반 학습 퀴즈 생성 AI입니다.
        아래 뉴스 내용을 참고하여 단답식 문제를 10개 만들어주세요.

        🎯 목표:
        - 뉴스의 핵심 사실(숫자, 인물, 정책명 등)을 직접 확인할 수 있는 문제

        ⚙️ 규칙:
        - 질문은 40자 이내, 한 문장
        - 정답은 한 단어 (뉴스에 명시된 내용만)
        - 명시된 내용이어도 시점의 경우 수치만 정답으로
        - 해설은 50자 이내, 사실적 근거 중심
        - JSON 배열로 출력

        📘 출력 형식(JSON 배열):
        [
          {{
            "question": "질문 내용",
            "answers": [
              {{
                "text": "답",
                "isCorrect": "true",
                "explanation": "정답에 대한 이유"
              }}
            ]
          }}
        ]

        뉴스 요약:
        {summary}
        """
        res = llm_i.invoke(prompt_i)
        text = res.content.strip().replace("```json", "").replace("```", "")
        return json.loads(text)

    # === E단계 문제 생성 ===
    def generate_advanced_e(i_quiz, summary):
        """I단계 10문항 중 5개를 자동 선정 후 E단계로 변환"""
        
        # === 1️. E단계 후보 선정 ===
        sel_prompt = f"""
        다음 I단계 문제 10개 중에서 E단계(고급 문어체)로 변환하기 적합한 5개를 선택하세요.
        - 정책·사회적 의미 중심
        - 단순 수치·날짜형 제외
        - JSON 배열로 질문만 출력 (예: ["질문1", "질문2", ...])

        문제 목록:
        {json.dumps([q["question"] for q in i_quiz], ensure_ascii=False, indent=2)}
        """
        sel_res = llm_i.invoke(sel_prompt)
        sel_text = sel_res.content.strip().replace("```json", "").replace("```", "")
        try:
            selected_questions = json.loads(sel_text)
        except json.JSONDecodeError:
            print("E단계 후보 선택 실패 — 랜덤으로 대체")
            selected_questions = random.sample([q["question"] for q in i_quiz], 5)

        # === 2️. 변환 대상 필터 ===
        target_items = [q for q in i_quiz if q["question"] in selected_questions]

        # === 3️. E단계 변환 ===
        prompt_e = f"""
        당신은 뉴스 기반 학습 설계자이자 교육 평가 전문가입니다.
        아래 추출된 I단계 문제 5개를 **고급 문어체 단답식(E 단계)** 문제 5개로 재작성하세요.

        🎯 목표:
        - I단계의 정답(answer)은 그대로 유지
        - 독해 속도를 늦추되, 사실적 근거는 기사에 명시된 내용만 사용
        - 질문은 최소 40자 이상

        🧠 해설 지침:
        - 해설은 단순히 ‘왜 정답인가’를 설명하는 수준을 넘어서야 함
        - 50자 내외로 아래 세 요소를 모두 포함해야 함:
          1. 정답의 사실적 근거
          2. 정답이 갖는 사회적·정책적 중요성
          3. 학습자가 이를 통해 얻는 인식 포인트

        ⚙️ 규칙:
        - question: 문어체, 35~45자
        - explanation: 최소 45자, 50자 내외
        - JSON 배열로 출력 (camelCase 필드명)
        - 기사 외 정보, 추측, 사견 절대 금지

        출력 예시:
        [
          {{
            "question": "국방부가 방첩사 개편 일정을 공식화한 시점은 언제인가?",
            "answers": [
              {{
                "text": "2026년",
                "isCorrect": "true",
                "explanation": "국방부는 방첩사 개편을 내년까지 완료하겠다고 명시했다. 이는 조직 재편의 실행력과 투명성을 학습자가 인식하도록 유도하기 위함이다."
              }}
            ]
          }}
        ]

        [E 문제 후보]
        {json.dumps(target_items, ensure_ascii=False, indent=2)}

        [뉴스 요약]
        {summary}
        """
        e_res = llm_e.invoke(prompt_e)
        text = e_res.content.strip().replace("```json", "").replace("```", "")

        try:
            e_items = json.loads(text)
        except json.JSONDecodeError:
            print("⚠️ JSON 파싱 실패 — LLM 응답 확인 필요")
            e_items = []

        # === 4️. 남은 I단계 5문항 ===
        remaining_i = [q for q in i_quiz if q["question"] not in selected_questions]

        print(f"변환된 E단계 문제 수: {len(e_items)}")
        print(f"남은 I단계 문제 수: {len(remaining_i)}")

        return e_items, remaining_i

    # === 실행 ===
    print("=== 뉴스 요약문 ===")
    print(summary)

    print("\n=== I단계 문제 생성 ===")
    i_quiz = generate_quiz_i(summary)
    print(json.dumps(i_quiz, ensure_ascii=False, indent=2))

    print("\n=== E단계 문제 변환 ===")
    e_quiz, remaining_i = generate_advanced_e(i_quiz, summary)

    print("\n[E단계 결과]")
    print(json.dumps(e_quiz, ensure_ascii=False, indent=2))

    print("\n[남은 I단계 5문항]")
    print(json.dumps(remaining_i, ensure_ascii=False, indent=2))

    # === 저장 ===
    SAVE_DIR = BASE_DIR / "data" / "quiz"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")

    final_result = [
        {
            "courseId": course_id,
            "sessionId": session_id,
            "topic": topic,
            "contentType": "short",
            "level": "i",
            "items": remaining_i
        },
        {
            "courseId": course_id,
            "sessionId": session_id,
            "topic": topic,
            "contentType": "short",
            "level": "e",
            "items": e_quiz
        }
    ]

    file_path = SAVE_DIR / f"{topic}_short_ie_{today}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    print(f"\n 전체 저장 완료 → {file_path.resolve()}")
    print("(I단계 5문항, E단계 5문항 — 총 10문항)")
    return final_result