import os, json, re
from datetime import datetime
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from quiz.select_session import select_session

def generate_ox_quiz(selected_session=None):
    # === 환경 변수 ===
    load_dotenv(override=True)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # === 세션 선택 ===
    if selected_session is None:
          selected_session = select_session()     
    topic = selected_session["topic"]
    course_id = selected_session["courseId"]            
    session_id = selected_session.get("sessionId")
    headline = selected_session.get("headline", "")
    summary = selected_session.get("summary", "")
    sourceUrl = selected_session.get("sourceUrl", "")

    print(f"\n선택된 태그: {course_id}")
    print(f"sessionId: {session_id}")
    print(f"제목: {headline}\n")

    # === 모델 ===
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # === 프롬프트 ===
    prompt_ox_n = f"""
    당신은 경제 뉴스 기반 학습용 퀴즈 생성 AI입니다.
    다음 뉴스 요약을 참고하여 **OX 퀴즈 5개 (N단계)**를 만들어주세요.

    [목표]
    - 입문자도 쉽게 풀 수 있는 **기초 사실 확인형(N단계)** 문제를 만듭니다.
    - 문장은 간결하고 명확하게, 기사 내용만으로 판단 가능해야 합니다.

    [규칙]
    1. 모든 문항은 명확히 O 또는 X로 판단 가능한 문장으로 작성합니다.
    2. 뉴스 요약의 사실만 사용하며, 새로운 정보·추측·예측은 금지합니다.
    3. 각 문항은 서로 다른 사실을 다뤄야 합니다.
    4. 문장은 '~했다', '~아니다'처럼 단정형으로 씁니다.
    5. 정답(O/X)은 다양하게 섞습니다.
    6. 해설은 한 줄(50자 이내)로, 사실 근거를 명확히 설명하는 -다로 끝나는 문장.

    7. 출력은 아래 JSON 형식만으로 반환합니다.

    출력 형식(JSON):
    [
    {{
        "contentId": 문제 번호,
        "question": "문장 형태의 문제",
        "correctAnswer": "정답 / O 또는 X 중 하나",
        "answerExplanation": "해설"
        "sourceUrl": "{sourceUrl}"
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

    # === 결과 출력 및 저장 ===
    print("\n=== 뉴스 요약문 ===\n")
    print(summary)
    print("\n=== 생성된 N단계 OX 퀴즈 ===\n")

    # === 변환: NIEdu 포맷 기반 ===
    ox_contents = []
    for i, q in enumerate(parsed_n, start=1):
        ans = q.get("correctAnswer", q.get("answer", "")).strip().upper()
        explanation = q.get("answerExplanation", "")
        question = q.get("question", "").strip()

        # 유효성 체크
        if ans not in ["O", "X"]:
            ans = "O"  # fallback

        ox_contents.append({
            "contentId": i,
            "question": question,
            "correctAnswer": ans,
            "answerExplanation": explanation,
            "sourceUrl": sourceUrl
        })

    # === NIEdu 포맷 ===
    result = {
        "contentType": "OX_QUIZ",  
        "level": "N",
        "contents": ox_contents
    }

    # === 콘솔 출력 ===
    for item in result["contents"]:
        print(f"{item['contentId']}. {item['question']}")
        print(f"정답: {item['correctAnswer']} | {item['answerExplanation']}\n")

    # === 파일 저장 ===
    BASE_DIR = Path(__file__).resolve().parents[2]
    SAVE_DIR = BASE_DIR / "data" / "quiz"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    file_path = SAVE_DIR / f"{topic}_{course_id}_{session_id}_OX_QUIZ_N_{today}.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {file_path.resolve()}")

#  실행
if __name__ == "__main__":
    generate_ox_quiz()