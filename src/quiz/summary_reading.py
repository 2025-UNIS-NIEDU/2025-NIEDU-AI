# src/summary_reading with keybert version
import os, json, re, logging
from datetime import datetime
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
from openai import OpenAI
from quiz.select_session import select_session

logger = logging.getLogger(__name__)

def generate_summary_reading_quiz(selected_session=None):

    load_dotenv(override=True)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)

    # === 세션 선택 ===
    if selected_session is None:
        selected_session = select_session()

    topic = selected_session["topic"]
    course_id = selected_session["courseId"]
    session_id = selected_session.get("sessionId")
    headline = selected_session.get("headline", "")
    summary = selected_session.get("summary", "")

    logger.info(f"[{topic}] 코스 {course_id} 세션 {session_id} SUMMARY_READING생성 시작 — 제목: {headline}")

    # === 요약문 정제 ===
    prompt = f"""
    너는 '뉴스 문해력 학습용 교재 요약문'을 교정하는 전문 편집자이다.
    문체는 간결하고 분석적이며 중립적이어야 한다.

    [핵심 규칙]
    1. 입력 요약문에 없는 사실, 날짜, 인물, 기관 정보를 새로 만들지 않는다.
    2. 감정적·비유적·추측성 표현(예: 논란, 비판적, 충격 등)은 모두 제거한다.
    3. 직함만 제시된 인물(예: '이 대통령')은 기사 제목에서 특정 가능한 경우에만 전체 이름/직함으로 치환한다.
    4. 날짜·연도·기간 관련 정보는 입력에 등장한 경우에만 그대로 사용하며, 입력에 없으면 절대 생성하지 않는다.  
    5. 직접 인용(" ")은 입력 문장에 실제 존재하는 내용만 사용하고, 그 외는 모두 자연스러운 간접 인용으로 바꾼다.  
    6. 모든 문장은 과거 시제로 통일한다.

    [가독성 규칙]
    6. 문장은 자연스럽게 이어지도록 필요한 범위에서 접속어나 연결 표현을 사용할 수 있다.
    7. 문장 구조는 ‘배경 → 발언 → 설명 → 정리’ 흐름으로 재배열할 수 있다. (사실 추가는 금지)

    [출력 형식]
    다음 JSON 형식으로만 출력한다.
    {{
    "summary": "<교정된 요약문 (200~250자)>"
    }}

    [기사 제목]
    {headline}

    [입력 요약문]
    {summary}
    """
    # === OpenAI Chat Completion 호출 ===
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 필요 시 "gpt-4o"나 "gpt-5"로 변경 가능
        messages=[
            {"role": "system", "content": "너는 뉴스 문해력 학습용 요약문을 교정하는 전문 편집자이다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    refined_summary_raw = response.choices[0].message.content.strip()

    # summary 값만 추출
    try:
        refined_summary_json = json.loads(re.sub(r"```json|```", "", refined_summary_raw))
        refined_summary = refined_summary_json.get("summary", "").strip()
    except Exception as e:
        logger.warning(f"JSON 파싱 실패(1차): {e}")
        refined_summary = refined_summary_raw  

    prompt_refine = f"""
    너는 '뉴스 문해력 학습용 교재'의 전문 편집자이다.

    아래 [입력 요약문]을 더 자연스럽고 매끄럽게 다듬어라.
    사실과 맥락은 그대로 유지하되, 문장 흐름만 정돈한다.

    [지침]
    1. 문장은 자연스럽고 읽기 쉽게 연결한다.
    2. 정보는 추가하지 않는다. (추론·창작 금지)
    3. 원문에 등장한 사실·인물·기관·지명만 사용한다.
    4. 문체는 중립적이고 보도체 톤으로 유지한다.
    5. 날짜·연도·기간은 입력에 있는 경우에만 그대로 사용하고, 새로 만들지 않는다.
    6. 전체 길이는 180~250자 사이로 맞춘다.

    [출력 형식]
    {{
    "summary": "<자연스럽고 논리적으로 정돈된 요약문>"
    }}

    [입력 요약문]
    {refined_summary}
    """

    # === OpenAI Chat Completion 호출 ===
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "너는 뉴스 문해력 학습용 요약문을 교정하는 전문 편집자이다."},
            {"role": "user", "content": prompt_refine}
        ],
    )

    refined_summary_raw = response.choices[0].message.content.strip()
    # summary 값만 추출
    try:
        refined_summary_json = json.loads(re.sub(r"```json|```", "", refined_summary_raw))
        refined_summary = refined_summary_json.get("summary", "").strip()
    except Exception as e:
        logger.warning(f"JSON 파싱 실패(2차): {e}")
        refined_summary = refined_summary_raw  

    # === 핵심 정답 추출 ===
    prompt_answer = f"""
    당신은 뉴스 요약문에서 Actor(주체)와 Object(핵심 개념)을 추출하는 정보 구조화 엔진입니다.

    [추출 대상 정의]

    1. Actor(주체)
    - 사건의 중심이 되는 인물·기관·단체 중 단 1개
    - 발언·결정·조치의 주체
    - 요약문에 등장한 고유명사만 선택

    2. Object(핵심 개념)
    - 사건 전체의 "핵심 논점" 또는 “중심 대상”인 명사/복합명사
    - 반드시 1개만 선택
    - 다음 중 하나여야 함:
    (a) 정책명
    (b) 문서명
    (c) 협상·절차명
    (d) 사건명
    - 반드시 요약문에 등장한 명사 
    - 동사형/조사 포함 금지 (예: '~하기로', '~했다')
    - Actor의 주장·비판에서 나온 별칭 금지 (예: '국익 시트')

    -----------------------------------------
    [선택 규칙]

    Object 선택 절차:
    1) 요약문에서 ‘정책/문서/협상/절차/사건명’ 후보만 추출  
    2) “이 사건은 무엇을 둘러싸고 벌어졌는가?” 기준으로 1개 선택  
    3) 문서명·정책명이 등장하면 절차명보다 우선한다  
    (예: '팩트시트' > '비준 절차')

    Actor 선택 절차:
    1) 가장 중심적으로 행동하거나 발언한 주체를 1개 선택  
    2) 집단(여야)도 가능하나, 특정 인물이 더 핵심이면 인물 우선  

    -----------------------------------------
    [출력 형식]

    아래 형식으로만 출력하라:

    {{
    "keywords": [
        {{"word": "ACTOR"}},
        {{"word": "OBJECT"}}
    ]
    }}

    -----------------------------------------
    뉴스 요약문:
    {refined_summary}
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_answer}],
        temperature=0
    )
    answers_json = json.loads(re.sub(r"```json|```", "", resp.choices[0].message.content.strip()))
    if isinstance(answers_json, dict) and "keywords" in answers_json:
        answers = [a["word"] for a in answers_json["keywords"]]
    elif isinstance(answers_json, list):
        answers = [a["word"] for a in answers_json]
    else:
        raise ValueError(f"예상치 못한 JSON 구조: {answers_json}")

    # === 3. LLM 혼동 가능성 평가 ===
    prompt_confuse = f"""
    다음 요약문에서 정답 단어 2개(Actor / Object)를 기준으로,
    요약문을 읽고 해당 내용 안의 **명사**들이 '오답으로 적합한지' 판단하고 score를 매겨라.

    [정답 역할]
    - 첫 번째 단어: 사건의 주체(Actor)
    - 두 번째 단어: 사건의 핵심 개념(Object)

    [오답 필터 규칙 — 아래 중 하나라도 걸리면 오답에서 제외]
    1) 형태 조건: 조사가 포함됨, 동사/형용사, 1~2글자, 명사가 아님
    2) 관계 조건: 정답과 동일/유사/포함/파생/축약
    3) 맥락 조건: 기사에서 중심 역할(Actor/Object)로 쓰인 단어

    [오답 선정 기준]
    - Actor형: 기사에 등장하지만 사건의 주체가 아닌 인물/조직/기관/국가/직책 등의 **명사**
    - Object형: 핵심 개념과 관련은 있으나 중심이 아닌 정책/제도/사건/보조 개념 등의 **명사**

    [출력 조건]
    - 최소 13개
    - actor/object 균형적으로 포함
    - 왜 학생이 혼동할 수 있는지 간단히 reason 명시

    [출력 형식]
    {{
    "ranked": [
        {{"word": "정부", "role": "actor", "score": 0.83, "reason": "요약문에 등장하지만 주체가 아님"}},
        {{"word": "정책", "role": "object", "score": 0.81, "reason": "핵심 개념과 관련 있으나 중심 아님"}}
    ]
    }}

    [입력]
    요약문:
    {refined_summary}

    정답:
    {answers}
    """
    resp_confuse = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt_confuse}],
        temperature=0
    )

    try:
        llm_ranked_data = json.loads(re.sub(r"```json|```", "", resp_confuse.choices[0].message.content.strip()))
    except json.JSONDecodeError as e:
        logger.error(f"[{topic}] JSON 파싱 오류(혼동어): {e}", exc_info=True)
        raise

    llm_ranked = {item["word"]: item["score"] for item in llm_ranked_data["ranked"]}
    filtered_combined=dict(llm_ranked)

    # === 6. 난이도별 분류 ===
    distractors = sorted(filtered_combined.items(), key=lambda x: x[1], reverse=True)[:9]
    while len(distractors) < 9:
        distractors.append(("기타", 0.0))

    e_level = distractors[0:3]
    i_level = distractors[3:6]
    n_level = distractors[6:9]

    # === 7. JSON 생성 ===
    def make_keyword_block(level, correct_list, distractors):
        level = level.upper()
        correct_two = correct_list[:2] if len(correct_list) >= 2 else correct_list
        summary_block = refined_summary

        keywords = [{"word": c, "isTopicWord": True} for c in correct_two]
        keywords += [{"word": d[0], "isTopicWord": False} for d in distractors[:3]]

        return {
            "contentType": "SUMMARY_READING",
            "level": level,
            "contents": [
                {
                    "summary": summary_block,
                    "keywords": keywords
                }
            ]
        }

    correct_list = answers[:2] if len(answers) >= 2 else answers
    final_json = [
        make_keyword_block("n", correct_list, n_level),
        make_keyword_block("i", correct_list, i_level),
        make_keyword_block("e", correct_list, e_level)
    ]

    logger.info(f"[{topic}] SUMMARY_READING JSON 변환 완료")

    # === 8. 파일 저장 ===
    BASE_DIR = Path(__file__).resolve().parents[2]
    QUIZ_DIR = BASE_DIR / "data" / "quiz"
    QUIZ_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")

    for block in final_json:
        level = block.get("level", "").upper().strip()
        if not level:
            continue
        file_path = QUIZ_DIR / f"{topic}_{course_id}_{session_id}_SUMMARY_READING_{level}_{today}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump([block], f, ensure_ascii=False, indent=2)
        logger.info(f"[{topic}] {level} 단계 SUMMARY_READING 저장 완료 → {file_path.name}")

    logger.info(f"[{topic}] 코스 {course_id} 세션 {session_id} SUMMARY_READING 퀴즈 생성 완료")

# === 실행 ===
if __name__ == "__main__":
    generate_summary_reading_quiz()