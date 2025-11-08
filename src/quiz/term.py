import os, re, json, requests, logging
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from quiz.select_session import select_session

logger = logging.getLogger(__name__)

def generate_term_quiz(selected_session=None):
    # === 1️. 환경 변수 로드 ===
    BASE_DIR = Path(__file__).resolve().parents[2]
    ENV_PATH = BASE_DIR / ".env"
    load_dotenv(ENV_PATH, override=True)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY")
    GOOGLE_CSE_CX_DICT = os.getenv("GOOGLE_CSE_CX_DICT")
    client = OpenAI(api_key=OPENAI_API_KEY)

    # === 2. 세션 선택 ===
    if selected_session is None:
        selected_session = select_session()

    topic = selected_session["topic"]
    course_id = selected_session["courseId"]            
    session_id = selected_session.get("sessionId")
    headline = selected_session.get("headline", "")
    summary = selected_session.get("summary", "")

    logger.info(f"[{topic}] 세션 {session_id} TERM_LEARNING 시작 — 제목: {headline}")

    # === 3️. 섹션별 전문용어 추출 템플릿 ===
    PROMPT_TEMPLATES = {
        "politics": "정치·법률 관련 제도, 법률, 정책, 절차, 기관명 등 4개 추출",
        "economy": "경제·금융 관련 제도, 지표, 정책, 용어, 금융상품 등 4개 추출",
        "society": "사회·교육·복지 관련 제도, 정책, 사회현상, 제도명 등 4개 추출",
        "world": "국제정치·외교·안보 관련 제도, 협정, 기구, 정책, 용어 등 4개 추출",
    }

    topic_key = next((t for t in PROMPT_TEMPLATES.keys() if t in topic.lower()), None)

    prompt = f"""
    당신은 뉴스 요약문에서 핵심 전문용어를 추출하고 표제어 형태로 정제하는 전문가입니다.
    아래 뉴스의 주제는 **{topic_key}**이며, 다음 기준에 따라 용어를 최대 4개 추출하세요.

    [용어 선정 기준]
    1. {PROMPT_TEMPLATES[topic_key]} 와 밀접하게 연관된 **전문적·기술적 개념**일 것  
    2. 뉴스 요약문에 **직접 등장하거나 암시된 핵심 주제**일 것  
    3. 일상적인 일반어는 제외하고, 해당 주제(도메인) 내에서만 사용되는 전문어(예: 제도명, 정책명, 기술명, 지표명 등)만 추출하세요.
    4. 각 용어는 1~3단어 이내의 **표제어 형태**로 정제  
    5. 불필요한 조사나 동사형(예: 추진, 논의, 강화)은 제거  
    6. 인명·기관명·지명·상표명은 제외  
    7. 의미 중복이 없도록 상호 구별되는 용어만 남김  

    [뉴스 요약문]
    {summary}
    """

    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw_terms = res.choices[0].message.content.strip()
    terms = re.split(r'[\n,]+|\d+\.\s*', raw_terms)
    terms = [t.strip() for t in terms if t.strip()]
    terms = terms[:4]  

    # === 4. LLM 기반 일반어 필터링 ===
    filter_prompt = f"""
    당신은 뉴스 요약문에서 추출된 후보 용어들을 평가하는 **정책·언론 전문 분석가**입니다.
    아래 용어 중 **일상적·포괄적 표현은 모두 제거**하고, 
    **정책·제도·외교·경제·법률 분야에서 공식적으로 사용되는 전문 용어만 남기세요.**

    [판단 기준]
    - **유지**: 
    - 공식 명칭(제도, 법률, 정책, 기구명, 회담명, 협정명 등)
    - 실제 기사·공문·보도자료에서 특정 사건·기구를 지칭하는 고유명사
    - 복합어라도 맥락이 특정 영역(외교, 경제, 산업 등)으로 한정될 때
    - **제거**:
    - 일반명사, 추상명사, 형용적 표현 (예: 논의, 계획, 관계, 협력, 개선, 방안, 제언)
    - 일상적 조합어 (예: 경제 협력, 사회 변화, 관계 개선)
    - 비구체적 단어 또는 맥락 없이 쓰이는 용어
    - **예시**
    - 유지: 한중정상회담, 한한령, 탄소국경조정제, 국정감사, 기준금리, 보조금법
    - 제거: 협력, 관계, 제언, 산업, 정책, 사회, 발전, 논의, 변화, 계획

    [출력 형식]
    전문 용어만 남긴 JSON 배열로 출력하세요. 예시:
    ["한중정상회담", "한한령", "탄소국경조정제"]

    [뉴스 요약문]
    {summary}

    [1차 추출 용어]
    {', '.join(terms)}
    """


    filter_res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": filter_prompt}],
        temperature=0
    )

    # --- JSON 파싱 시도 ---
    try:
        filtered_terms = json.loads(filter_res.choices[0].message.content.strip())
    except Exception as e:
        logger.warning(f"[{topic}] 필터링 JSON 파싱 오류: {e}")
        filtered_terms = terms  

    # ===  최소 2개 이상 보장 로직 추가 ===
    if not filtered_terms or len(filtered_terms) < 2:
        logger.warning(f"[{topic}] 필터링 결과 2개 미만 → 원본 일부 보강")
        needed = 2 - len(filtered_terms)
        for t in terms:
            if t not in filtered_terms:
                filtered_terms.append(t)
            if len(filtered_terms) >= 2:
                break

    # --- 필터링 결과 반영 ---
    if filtered_terms:
        terms = filtered_terms
    else:
        logger.warning(f"[{topic}] 필터링 후 용어 없음 → 원본 사용")
        terms = terms  

    # --- 중간 점검 (필터링 이후 실행) ---
    logger.info(f"[{topic}] 최종 전문용어: {terms}")

    # === 5️ 용어 정의  ===
    def fetch_definition(term):
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CSE_CX_DICT,
            "q": f"{term} 정의 OR 의미",
            "num": 10,
            "lr": "lang_ko"
        }

        res = requests.get(url, params=params)
        if res.status_code != 200:
            logger.warning(f"[{term}] Google 검색 실패 (status={res.status_code})")
            return term, None

        data = res.json()
        items = data.get("items", [])
        if not items:
            return term, None

        best_item = max(
            items,
            key=lambda x: len(x.get("snippet", "")) if x.get("snippet") else 0
        )

        title = best_item.get("title", "")
        snippet = best_item.get("snippet", "")
        snippet = snippet.strip() if snippet else None
        return term, snippet

    # === 5.5 용어 정의 완성 ===
    def complete_snippet(term, snippet, summary):
        prompt = f"""
    당신은 '{term}'의 개념을 뉴스 요약문 맥락에서 자연스럽고 정확하게 정의하는 전문가입니다.

    [뉴스 요약문]
    {summary}

    [입력된 정의 후보]
    {snippet}

    [작성 규칙]
    1. 먼저 위 문장이 '{term}'의 정의로서 뉴스 요약문 맥락에 **적절한지** 판단하세요.
    2. 만약 문장이 맥락과 다르거나, 사실상 의미가 어긋나거나, 지나치게 일반적이면 — '{term}'의 정의를 **새롭게 작성**하세요.
    3. 자연스러운 경우에는 문법과 어투만 다듬어 완전한 문장으로 정리하세요.
    4. 문체는 사전식 정의체 (“~이다.”, “~을 의미한다.”, “~을 말한다.” 등)로 마무리합니다.
    5. 불필요한 영어·기호(…, :, ·, -, “”)는 모두 제거합니다.
    6. 한 문장, 60자 이내로만 작성합니다.
    7. 정의의 초점은 **이 뉴스에서의 의미**에 두세요 (사전 일반 정의가 아님).

    출력은 정의 문장 한 줄만 하세요.
    """
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return res.choices[0].message.content.strip()

    # === 6️. 예시 문장 + 비유 생성 ===
    def build_examples(term, news_text):
        prompt_1 = f"""
        다음 뉴스 내용을 참고하여 "{term}"이(가) 포함된 자연스럽고 완전한 한 문장을 만들어주세요.
        - 실제 뉴스 문체(보도체)를 유지하세요.
        - 예시 문장끼리 겹치지 않게 다양하게 표현하세요.
        - 문체나 어휘가 다른 용어의 문장과 **중복되지 않도록** 다양하게 표현
        - 100자 이내로 1문장만 출력하세요.
        - -다.로 끝나는 단정형 문체 사용

        [뉴스 요약문]
        {news_text}
        """
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_1}],
            temperature=0.3
        )
        example = res.choices[0].message.content.strip()

        prompt_2 = f"""
        "{term}"을(를) 일상적 비유로 100자 이내로 간단히 설명하세요.
        - 최대 120자.
        - 전문용어 피하고, 직관적으로 이해 가능한 예시 사용
        - 다른 용어의 비유와 중복되지 않게 새로운 비유를 제시
        - 자연스러운 구어체,존댓말로 작성
        """
        res2 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_2}],
            temperature=0.7
        )
        analogy = res2.choices[0].message.content.strip()
        if len(analogy) > 120:
            analogy = analogy[:117] + "…"
        return example, analogy

    # === 7. LLM 결과 정리 ===
    if not terms or len(terms) == 0:
        llm_response = res.choices[0].message.content
        raw_terms = llm_response.strip()
        terms = re.split(r'[\n,]+|\d+\.\s*', raw_terms)
        terms = [t.strip() for t in terms if t.strip()]

    # === 8. 용어별 카드 생성 ===
    results = []
    for i, term in enumerate(terms, start=1):
        term, snippet = fetch_definition(term)
        completed = complete_snippet(term, snippet, summary) if snippet else None
        example, analogy = build_examples(term, summary)

        results.append({
            "termId": i,  
            "name": term,
            "definition": completed or snippet or "",
            "exampleSentence": example or "",
            "additionalExplanation": analogy or "",
        })

    # === 9. NIEdu 포맷 ===
    term_card = {
        "contentType": "TERM_LEARNING",  
        "level": "N",
        "contents": [
            {
            "terms": results  
            }
        ]
    }

    # === 10. 결과 출력 및 저장 ===
    logger.info(f"[{topic}] TERM_LEARNING 변환 완료 ({len(results)}개 용어)")

    QUIZ_DIR = BASE_DIR / "data" / "quiz"
    QUIZ_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    save_path = QUIZ_DIR / f"{topic}_{course_id}_{session_id}_TERM_LEARNING_N_{today}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(term_card, f, ensure_ascii=False, indent=2)

    logger.info(f"[{topic}] 저장 완료 → {save_path.name}")

# === 실행 ===
if __name__ == "__main__":
    generate_term_quiz()