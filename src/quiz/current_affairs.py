import os, re, json, requests, logging
from openai import OpenAI
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from datetime import datetime
from dotenv import load_dotenv
from quiz.select_session import select_session   

logger = logging.getLogger(__name__)

def generate_current_affairs_quiz(selected_session=None):
    """N단계 시사 이슈형 퀴즈 자동 생성"""

    # === 1. 환경 변수 로드 ===
    load_dotenv(override=True)

    GOOGLE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY")
    GOOGLE_NEWS_ID = os.getenv("GOOGLE_CSE_CX_NEWS")
    GOOGLE_GOV_ID = os.getenv("GOOGLE_CSE_CX_GOV")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)

    # === 2. 세션 불러오기 ===
    if selected_session is None:
        selected_session = select_session()     

    topic = selected_session["topic"]
    course_id = selected_session["courseId"]            
    session_id = selected_session.get("sessionId")
    headline = selected_session.get("headline", "")
    summary = selected_session.get("summary", "")

    logger.info(f"[{topic}] 세션 {session_id} 시사 이슈형 퀴즈 생성 시작")

    # === 3️. CSE 검색 함수 ===
    def get_cse_snippets(query, cx_id, n=10):
        url = "https://www.googleapis.com/customsearch/v1"
        params = {"q": query, "key": GOOGLE_API_KEY, "cx": cx_id, "num": n}
        res = requests.get(url, params=params)
        if res.status_code != 200:
            logger.warning(f"❌ CSE API 오류 ({res.status_code}) — {query}")
            return []
        data = res.json()
        snippets = []
        for item in data.get("items", []):
            snippets.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link": item.get("link", "")
            })
        return snippets

    # === 4️. 뉴스 + 정부 스니펫 수집 ===
    news_snippets = get_cse_snippets(selected_session["headline"], GOOGLE_NEWS_ID, n=10)
    gov_snippets = get_cse_snippets(selected_session["headline"], GOOGLE_GOV_ID, n=10)
    snippets = news_snippets + gov_snippets
    logger.info(f"[{topic}] 스니펫 수집 완료 — 뉴스 {len(news_snippets)} + 정부 {len(gov_snippets)}")

    # === 5️. 스니펫 병합 ===
    context = "\n\n".join(
        [f"[{i+1}] {s['title']}\n{s['snippet']}\nURL: {s['link']}" for i, s in enumerate(snippets)]
    )

    # === 6️. 프롬프트 구성 (절대 수정 금지) ===
    prompt_background_n = f"""
    당신은 '뉴스 문해력 학습용 분석 전문가'입니다.  

    ### 작성 원칙
    1. **인과관계의 4단 세트(원인→전개→결과→의미)**가 자연스럽게 이어지도록 작성하세요.  
    - 문장 간 논리 연결이 명확해야 하며, 각각은 독립적으로 존재하지 않고 앞뒤가 인과적으로 이어져야 합니다.
    - 문장 사이에는 반드시 인과 연결어를 사용하세요.
        - 원인: “~로 인해”, “~때문에”, “~을 계기로”
        - 전개: “이 과정에서”, “이에 따라”, “이를 둘러싸고”
        - 결과: “결국”, “그 결과”, “~으로 이어졌다”
        - 의미: “이는 ~을 의미한다”, “~한 시사점을 남겼다”, “~의 중요성을 일깨웠다”

    [항목별 지침]
    1. **issue (핵심 쟁점 요약)**  
    - 40자 이내, 주제와 논점이 한눈에 들어오도록.  
    - 단순 사건명이 아니라 “논란/갈등/정책 변화” 등 사건의 성격이 드러나야 함.

    2. **cause (사건의 배경 및 촉발 요인)**  
    - 100~120자.  
    - 사건이 발생한 제도적·정치적·사회적 배경을 구체적으로 기술.
    - 단순한 계기가 아니라, ‘왜 이 사건이 문제로 부상했는가’를 설명.  
    - 인물의 행위·제도적 맥락·정치적 배경을 구체적으로 명시.

    3. **circumstance (전개 과정 및 핵심 반응)**  
    - 100~120자.  
    - 시점별 주요 행위자(정부·정당·기관 등)의 입장과 반응을 비교 중심으로 기술.  
    - 단순한 반응 나열이 아니라, **행위자 간의 상호작용 구조**가 드러나야 한다.
    - 만약 요약문에 행위자·이유·배경 등 핵심 정보가 누락되어 공백이 생긴다면,
        기사에 드러난 맥락을 활용해 논리적으로 자연스러운 연결 문장으로 복원하라.

    4. **result (즉각적 결과·정치적 파장)**  
    - 80~100자.  
    - 논란 이후의 실질적 변화, 여론·정치권 반응·조치 등을 중심으로 기술.
    - 여론·정책 변화·조치 등 구체적 후속 반응을 중심으로 기술.

    5. **effect (장기적 영향·사회적 함의)**  
    - 80~100자.  
    - 제도적 신뢰, 정치 윤리, 사회 구조 등 보다 근본적 차원의 시사점을 제시.
    - 단순한 영향이 아니라 “이 사건이 사회에 던진 질문”이나 “재발 방지·제도 개선의 필요성” 등 학습적 통찰로 마무리한다.

    [작성 규칙]
    - 중복 표현 금지, 논리적 연결 유지.  
    - **추측 금지. 기사 또는 스니펫에 언급된 근거만 활용.**  
    - 출력은 반드시 **JSON 형식**으로만 작성할 것.

    [출력 예시]
    {{
    "CURRENT_AFFAIRS": {{
        "issue": "법제처장 '무죄 발언' 논란, 공직 중립성 논의 확산",
        "cause": "조원철 법제처장이 국회 답변 과정에서 특정 정치인의 혐의에 대해 '무죄라고 생각한다'는 발언을 하며 공직자의 정치적 중립성 논란이 촉발되었다. 이는 최근 정부기관의 정치적 발언이 잇따른 상황에서 발생해 논란이 증폭됐다.",
        "circumstance": "이 과정에서 여당은 법제처장의 발언이 부적절하다고 지적하며 사퇴를 요구했고, 야당은 이를 과도한 정치 공세로 규정하며 맞섰다. 양측의 공방이 이어지면서 국회 내 논의가 정치적 공방으로 확산됐다.",
        "result": "결국 법제처장은 공식 사과문을 발표하고 발언 경위를 해명했으나, 공직자의 발언 기준에 대한 논쟁은 계속됐다.",
        "effect": "이번 논란은 공직자의 정치적 중립성과 발언 책임의 중요성을 다시금 환기시켰다. 향후 정부 기관의 발언 기준을 명확히 할 제도 개선 필요성이 제기됐다."
    }}
    }}

    ---
    [기사 요약문]
    {selected_session.get("summary")}

    [관련 검색 스니펫 (총 {len(snippets)}개)]
    {context}
    """

    # === 7️. LLM 호출 ===
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_background_n}],
            temperature=0.3
        )
        raw_output = resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"[{topic}] LLM 호출 실패: {e}", exc_info=True)
        return

    # === 8. JSON 파싱 ===
    try:
        cleaned = re.sub(r"```json|```", "", raw_output)
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(f"[{topic}] JSON 파싱 실패 — 원문 그대로 저장")
        parsed = {
            "CURRENT_AFFAIRS": {
                "issue": "",
                "cause": "",
                "circumstance": "",
                "result": "",
                "effect": ""
            }
        }

    # === 구조 검증 ===
    affairs = parsed.get("CURRENT_AFFAIRS", {})
    answers = {
        "issue": affairs.get("issue", ""),
        "cause": affairs.get("cause", ""),
        "circumstance": affairs.get("circumstance", ""),
        "result": affairs.get("result", ""),
        "effect": affairs.get("effect", "")
    }

    # === 최종 래핑 및 저장 ===
    wrapped_output = {
        "contentType": "CURRENT_AFFAIRS",
        "level": "N",
        "contents": [answers],
    }

    BASE_DIR = Path(__file__).resolve().parents[2]
    SAVE_DIR = BASE_DIR / "data" / "quiz"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    file_path = SAVE_DIR / f"{topic}_{course_id}_{session_id}_CURRENT_AFFAIRS_N_{today}.json"

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(wrapped_output, f, ensure_ascii=False, indent=2)
        logger.info(f"[{topic}] CURRENT_AFFAIRS 저장 완료 → {file_path.name}")
    except Exception as e:
        logger.error(f"[{topic}] 파일 저장 실패: {e}", exc_info=True)

# === 실행 ===
if __name__ == "__main__":
    generate_current_affairs_quiz()