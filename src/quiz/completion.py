import os, json, random, logging
from datetime import datetime
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from quiz.select_session import select_session

logger = logging.getLogger(__name__)

def generate_completion_quiz(selected_session=None):
    """E단계 문장 완성형 퀴즈 자동 생성"""
    
    # === 환경 변수 로드 ===
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

    logger.info(f"[{topic}] 세션 {session_id} 문장 완성형 퀴즈 생성 시작")

    # === 모델 설정 ===
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

    def generate_sentence_completion_quiz(summary: str):
        prompt = f"""
[목표]
1. 뉴스의 핵심 내용을 완성하도록 유도합니다.
2. 학습자가 전체 문맥을 이해해야 자연스럽게 완성할 수 있어야 합니다.
3. 정답은 기사 요약의 사실에 기반한 완전한 한 문장으로 작성합니다.
4. 해설은 생성하지 않습니다.

[규칙]
1. question:
  - 미완성 문장 형태로, 약 25~40자 이내
  - 문장의 핵심 정보(원인·결과·주체 중 하나)가 빠진 상태여야 함
  - 마지막은 반드시 ‘______’ 또는 문장 중간에서 끊긴 불완전한 형태로 끝냄
2. referenceAnswer:
  - question을 완성하는 **단일 핵심 정보 중심의 한 문장**
  - 불필요한 이유절(예: "~을 위해", "~로 인해")은 생략
  - 30~50자 이내의 자연스러운 문장으로 작성
3. 세 문항은 서로 다른 정보 요소를 다루어야 함
  (예: 주체, 행위, 평가 등)
4. 기사에 직접 언급된 사실만 사용하며, 추측·감정·평가 표현 금지
5. JSON 배열 형태로 출력하며, 다른 텍스트나 설명은 포함하지 마세요.

출력 예시(JSON):
[
  {{
    "contentId": "문제 번호",
    "question": "질문",
    "referenceAnswer": "정답 문장"
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
    try:
        e_quiz = generate_sentence_completion_quiz(summary)
        if not e_quiz:
            logger.warning(f"[{topic}] 퀴즈 결과가 비어 있음 (session {session_id})")
            return
    except Exception as e:
        logger.error(f"[{topic}] 퀴즈 생성 실패: {e}", exc_info=True)
        return

    # === 저장 ===
    BASE_DIR = Path(__file__).resolve().parents[2]
    SAVE_DIR = BASE_DIR / "data" / "quiz"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")

    final_result = {
        "contentType": "SENTENCE_COMPLETION",
        "level": "E",
        "contents": [e_quiz],
    }

    file_path = SAVE_DIR / f"{topic}_{course_id}_{session_id}_SENTENCE_COMPLETION_E_{today}.json"
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        logger.info(f"[{topic}] E단계 퀴즈 저장 완료 → {file_path.name} ({len(e_quiz)}문항)")
    except Exception as e:
        logger.error(f"[{topic}] 파일 저장 실패: {e}", exc_info=True)

# === 실행 ===
if __name__ == "__main__":
    generate_completion_quiz()