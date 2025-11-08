import os, json, logging
from datetime import datetime
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from quiz.select_session import select_session

logger = logging.getLogger(__name__)

def generate_reflect_quiz(selected_session=None):
    # 환경 변수 및 경로 설정
    load_dotenv(override=True)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    BASE_DIR = Path(__file__).resolve().parents[2]
    QUIZ_DIR = BASE_DIR / "data" / "quiz"
    today = datetime.now().strftime("%Y-%m-%d")

    # 1️. 세션 선택
    if selected_session is None:
        selected_session = select_session()  

    topic = selected_session["topic"]
    courseId = selected_session["courseId"]
    sessionId = selected_session["sessionId"]

    logger.info(f"[{topic}] 세션 {sessionId} 리플렉션 생성 시작")

    # 2️. 선택된 세션 기반으로 파일 자동 필터링
    prefix = f"{topic}_{courseId}_{sessionId}_"
    target_files = sorted([f for f in QUIZ_DIR.glob(f"{prefix}*.json")])

    if not target_files:
        logger.warning(f"[{topic}] 관련 JSON 파일이 없습니다 ({prefix})")
        return

    # 3️. 모든 관련 파일 로드
    all_blocks = []
    for file_path in target_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_blocks.extend(data if isinstance(data, list) else [data])
        except json.JSONDecodeError:
            logger.warning(f"[{topic}] JSON 오류: {file_path.name}")

    # 4️. I/E 단계 분리
    i_contents, e_contents = [], []
    for block in all_blocks:
        if not isinstance(block, dict):
            continue
        level = (block.get("level", "")).upper()
        contents = block.get("contents", [])
        valid_questions = [c for c in contents if isinstance(c, dict) and c.get("question")]

        if level == "I":
            i_contents.extend(valid_questions)
        elif level == "E":
            e_contents.extend(valid_questions)
        else:
            logger.debug(f"인식 불가 level: {level}, 파일: {block.get('contentType')}")

    logger.info(f"[{topic}] I단계 {len(i_contents)}개, E단계 {len(e_contents)}개 로드 완료")

    # 6️. LLM 설정
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # 7️. 회고형 문제 생성 함수
    def generate_reflection(level, quiz_list):
        if level == "i":
            phase_desc = (
                "이해 및 분석 중심의 회고형 질문을 만드세요. "
                "뉴스의 사실과 인과관계를 되짚으며, 사건의 본질이나 의도를 탐구하는 데 초점을 둡니다. "
                "학습자가 사건의 의미를 스스로 분석하도록 유도해야 합니다."
            )
        else:
            phase_desc = (
                "비판적·확장적 사고를 유도하는 회고형 질문을 만드세요. "
                "사건이 사회, 제도, 가치, 윤리 등에 미치는 영향을 고려하고, "
                "다른 시각에서 재해석하거나 대안을 고민하도록 이끕니다."
            )

        prompt = f"""
    당신은 뉴스 기반 학습 퀴즈 생성 AI입니다.

    뉴스의 내용을 회고적으로 이해하도록 돕는 **{level.upper()}단계 회고형 질문 1개**를 만드세요.
    {phase_desc}

    아래에는 이미 생성된 퀴즈 목록이 주어집니다.
    이들을 참고해 “이 사건에 대해 어떤 점을 되돌아봐야 하는가”에 초점을 맞추세요.

    규칙:
    - 단문 문어체 (약 20자 내외)
    - 질문에는 반드시 사건의 맥락이 암시되어야 함
    - {("원인·의도·의미 중심" if level == "i" else "가치·함의·대안 중심")}
    - 반드시 JSON 형식으로 출력 (배열 형태)
    - 예시:
    [{{"question": "{'정부 개편안의 추진 배경은 무엇인가?' if level == 'i' else '정부 개편안이 사회적 신뢰에 미친 영향은?'}"}}]

    === {level.upper()}단계 문제 목록 ===
    {json.dumps(quiz_list, ensure_ascii=False, indent=2)}
    """
        res = llm.invoke(prompt)
        text = res.content.strip().replace("```json", "").replace("```", "")
        return json.loads(text)

    # 9.️ 회고형 문제 생성
    try:
        i_reflection = generate_reflection("i", i_contents)
        e_reflection = generate_reflection("e", e_contents)
    except Exception as e:
        logger.error(f"[{topic}] 리플렉션 생성 중 오류: {e}", exc_info=True)
        return

    # === 10. 결과 저장 ===
    def save_reflection_json(topic, courseId, sessionId, i_reflection, e_reflection):
        BASE_DIR = Path(__file__).resolve().parents[2]
        QUIZ_DIR = BASE_DIR / "data" / "quiz"
        QUIZ_DIR.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")

        final_result = [
            {"contentType": "SESSION_REFLECTION", "level": "I", "contents": i_reflection},
            {"contentType": "SESSION_REFLECTION", "level": "E", "contents": e_reflection},
        ]

        for item in final_result:
            level = str(item.get("level", "")).upper().strip()  
            if not level:
                continue
            file_name = f"{topic}_{courseId}_{sessionId}_SESSION_REFLECTION_{level}_{today}.json"
            file_path = QUIZ_DIR / file_name
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(item, f, ensure_ascii=False, indent=2)
                logger.info(f"[{topic}] {level}단계 리플렉션 저장 완료 → {file_path.name}")
            except Exception as e:
                logger.error(f"[{topic}] {level}단계 저장 오류: {e}", exc_info=True)

    save_reflection_json(topic, courseId, sessionId, i_reflection, e_reflection)
    logger.info(f"[{topic}] 세션 {sessionId} 리플렉션 생성 완료")

#  실행
if __name__ == "__main__":
    generate_reflect_quiz()