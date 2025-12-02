# === 표준 라이브러리 ===
import os, json, time, logging
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime

logger = logging.getLogger(__name__)

def refine_course_structure():
    """뉴스 학습 코스 정제 파이프라인 전체 실행"""

    today = datetime.now().strftime("%Y-%m-%d")

    # === 경로 설정 ===
    BASE_DIR = Path(__file__).resolve().parents[2]
    ENV_PATH = BASE_DIR / ".env"
    COURSE_DIR = BASE_DIR / "data" / "course_db"
    FILTER_DIR = COURSE_DIR / "filtered"
    FILTER_DIR.mkdir(parents=True, exist_ok=True)

    # === 환경 변수 로드 ===
    load_dotenv(ENV_PATH, override=True)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # === 학습용 코스 선별 프롬프트 ===
    PROMPT_SIMPLE_FILTER = """
    너는 뉴스 학습 코스의 편집자이다.
    아래에 제시된 **코스명과 세션(헤드라인) 전체를 하나의 세트로 보고**,  
    그 세트 전체가 '학습용 주제로서 타당한가'를 평가하라.

    ---

    ### 판정 기준

    너무 엄격하게 판단하지 말고, **전체적으로 사회적·정책적 의미가 있다면 True로 본다.**  
    아래의 기준은 참고용이며, **하나라도 사회적 성찰, 제도, 정책, 변화의 맥락이 있다면 True로 판단**한다.

    ---

    #### True (학습용으로 적절)
    - headline들이 제도, 정책, 사회 문제, 산업 변화, 기술 발전 등 구조적 주제를 다룸  
    - 서로 연결된 흐름이나 사회 변화의 원인-결과 관계가 있음  
    - 인물 중심 기사라도 발언·행동이 정책, 제도, 사회적 이슈와 관련 있음  
    - 다소 평범하더라도 **사회적 맥락 속에서 사고를 확장할 수 있다면 유지**

    ---

    #### False (정말 학습용으로 부적절)
    아래 중 **하나에만 해당되어도 False로 판단**한다.

    - headline들이 대부분 피상적·가십성 기사  
    - 개인 사건, 범죄, 재난, 화재, 날씨 등 단발성 보도  
    - 기업 홍보, 신제품 출시, 축제·시상식 등 상업성 내용  
    - 감정적 대립, 비난 중심의 자극적 기사로 구조적 의미 없음  
    - headline 간 주제 일관성 전혀 없음

    ---

    ### 출력 형식 (JSON)
    {"is_educational": true or false, "reason": "한 문장 설명"}
    """

    # === 안전한 요청 ===
    def safe_request(prompt, retry=3, wait=3):
        for attempt in range(retry):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )
                return resp.choices[0].message.content
            except Exception as e:
                time.sleep(wait)
        return None

    # === 부적절 코스 2차 검증 ===
    def double_check_false(course, first_reason):
        """
        2차 검증: 1차에서 False로 판정된 코스를 다시 확인하되,
        진짜로 정책·제도·사회 구조적 맥락이 없으면 그대로 False 유지.
        """
        summaries = "\n".join([f"- {s.get('summary', '')}" for s in course.get("sessions", []) if s.get("summary")])
        prompt = f"""
    너는 뉴스 학습 코스의 검증자이다.
    이 코스는 1차에서 '학습용으로 부적절(False)'하다고 판단되었다.
    이번에는 그 판단이 **정당한지 다시 검증**하라.

    ---

    ### 검증 원칙
    - summary들이 **정책, 제도, 사회 구조, 산업 변화, 기술 발전, 국제 관계** 등
    **구조적 변화나 제도적 시사점을 다룬 경우에만 True**로 변경한다.
    - summary들이 **축제, 문화행사, 시상식, 홍보, 인터뷰, 단순 소감** 중심이라면
    **False를 유지한다.**
    - 단순히 ‘공동체’, ‘문화’, ‘소통’, ‘참여’ 같은 단어가 포함되어 있다고 해서
    True로 바꾸지 마라.  
    (이것은 정책·제도적 맥락이 아니면 학습용 주제로 보기 어렵다.)
    - summary 중 과반수 이상이 **정책 변화, 제도 논의, 사회적 구조 개혁, 갈등 해결 등**
    구체적 맥락이 있으면 True로 바꿔도 된다.

    ---

    이전 판단 사유:
    {first_reason}

    코스명: {course.get("courseName")}
    요약문 목록:
    {summaries}

    출력 형식(JSON):
    {{"is_educational": true or false, "reason": "한 문장 설명"}}
    """
        resp = safe_request(prompt)
        if not resp:
            return {"is_educational": False, "reason": "2차 검증 실패"}

        try:
            parsed = json.loads(clean_json_response(resp))
            return parsed
        except:
            return {"is_educational": False, "reason": "2차 파싱 실패"}

    # === JSON 파싱 오류 방지 ===
    def clean_json_response(resp: str) -> str:
        return (
            resp.strip()
            .replace("```json", "")
            .replace("```", "")
            .replace("`", "")
            .strip()
        )
    
    # === 헤드라인 기반 세션 선택 프롬프트 ===
    PROMPT_SELECT_SESSIONS_TEMPLATE = """
    너는 뉴스 학습 코스의 편집자이다.

    아래에 코스명과 해당 코스에 포함된 세션들의 headline 목록이 주어진다.
    너의 역할은 **코스명과 의미적으로 가장 강하게 연관된 headline 5개만 선택하는 것**이다.

    ---

    ### 선택 기준 (중요)

    - headline만 보고 판단한다. (summary는 고려하지 않는다)
    - 코스명과 주제적으로 직접 연결되는 기사 우선
    - 코스명 주요 키워드가 headline에 있음, 또는 의미적으로 연관되면 선택
    - 단순 문자열 매칭이 아니라 headline 전체 의미 기반으로 판단
    - 정치/경제/사회/국제적 맥락까지 고려해서 의미적 연관성 중심으로 선택

    ---

    ### 출력 형식(JSON)
    {{
    "selected_sessions": [
        {{ "index": 3 }},
        {{ "index": 5 }},
        {{ "index": 1 }},
        {{ "index": 7 }},
        {{ "index": 2 }}
    ],
    "reason": "코스명과 가장 직접적으로 연결된 headline들을 선택함"
    }}

    ### 코스명
    {courseName}

    ### headline 목록
    {session_list}
    """

    def select_top5_sessions(client, course):
        """코스명과 headline 의미적 연관성을 기준으로 LLM이 세션 5개 선택"""

        # numbering 붙여서 텍스트로 정리
        session_list_txt = "\n".join([
            f"{i+1}. {s.get('headline', '')}"
            for i, s in enumerate(course["sessions"])
        ])

        prompt = PROMPT_SELECT_SESSIONS_TEMPLATE.format(
            courseName=course.get("courseName"),
            session_list=session_list_txt,
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        ).choices[0].message.content

        try:
            parsed = json.loads(
                resp.replace("```json", "").replace("```", "").strip()
            )
            idxs = [x["index"] - 1 for x in parsed["selected_sessions"]]
            return [course["sessions"][i] for i in idxs]

        except:
            # 실패하면 fallback
            return course["sessions"][:5]

    # === 메인 정제 함수 ===
    def refine_course_simple(topic: str):

        topic_path = COURSE_DIR / f"{topic}_{today}.json"
        if not topic_path.exists():
            return

        with open(topic_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        refined_courses = []

        for c in tqdm(data, desc=f"[{topic}] 1단계: LLM 필터링"):
            headlines = "\n".join([f"- {s.get('headline', '')}" for s in c.get("sessions", [])])
            prompt = f"{PROMPT_SIMPLE_FILTER}\n\n코스명: {c.get('courseName')}\n{headlines}"

            resp = safe_request(prompt)
            if not resp:
                continue

            resp_clean = clean_json_response(resp)

            try:
                parsed = json.loads(resp_clean)
            except:
                continue

            if parsed.get("is_educational") is False:
                recheck = double_check_false(c, parsed.get("reason", "사유 없음"))

                if recheck.get("is_educational") is False:
                    continue

            # === 최종 세션 5개로 제한 ===
            c["sessions"] = select_top5_sessions(client, c)

            def parse_datetime(x):
                try:
                    return datetime.fromisoformat(x.get("publishedAt"))
                except:
                    return datetime.min  # 날짜 없을 때 대비

            c["sessions"].sort(key=parse_datetime, reverse=True)

            # 정렬 후 sessionId 다시 부여
            for idx, s in enumerate(c["sessions"], start=1):
                s["sessionId"] = idx

            refined_courses.append(c)

        # === 인덱스 정리 ===
        for i, c in enumerate(refined_courses, 1):
            c["courseId"] = i
            for j, s in enumerate(c["sessions"], 1):
                s["sessionId"] = j

        out_path = FILTER_DIR / f"{topic}_{today}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(refined_courses, f, ensure_ascii=False, indent=2)
        
    # === 모든 토픽 자동 정제 ===
    TOPICS = ["politics", "economy", "society", "world"]
    for topic in TOPICS:
        try:
            refine_course_simple(topic)
        except Exception as e:
            pass

# === 실행 ===
if __name__ == "__main__":
    refine_course_structure()