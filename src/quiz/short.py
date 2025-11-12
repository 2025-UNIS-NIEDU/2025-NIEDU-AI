import os, sys, json, yaml, logging, random
from datetime import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
from keybert import KeyBERT
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from quiz.select_session import select_session

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def generate_short_quiz(selected_session=None):
    """뉴스 기반 단답형 퀴즈 자동 생성 (I/E 단계, 요약문 기반)"""

    # === 1. 환경 변수 ===
    BASE_DIR = Path(__file__).resolve().parents[2]
    ENV_PATH = BASE_DIR / ".env"
    load_dotenv(ENV_PATH, override=True)

    # === 2. 세션 선택 ===
    if selected_session is None:
        selected_session = select_session()

    topic = selected_session["topic"]
    course_id = selected_session["courseId"]
    session_id = selected_session.get("sessionId")
    sourceUrl = selected_session.get("sourceUrl", "")

    logger.info(f"\n[INFO] {topic} 코스 {course_id} 세션 {session_id} SHORT_ANSWER 생성 시작")

    # === SUMMARY_READING 불러오기 ===
    SUMMARY_DIR = BASE_DIR / "data" / "quiz"
    today = datetime.now().strftime("%Y-%m-%d")
    pattern = f"{topic}_{course_id}_{session_id}_SUMMARY_READING_*_{today}.json"
    summary_files = sorted(SUMMARY_DIR.glob(pattern))

    if not summary_files:
        logger.warning("요약문 파일을 찾을 수 없습니다. select_session의 기본 summary 사용.")
        summary = selected_session.get("summary", "")
    else:
        with open(summary_files[-1], "r", encoding="utf-8") as f:
            data = json.load(f)
        try:
            content = data[0]["contents"][0]
            summary = content["summary"]
            logger.info(f"요약문 불러옴 → {summary_files[-1].name}")
        except Exception as e:
            logger.warning(f"요약문 파싱 실패: {e}")
            summary = selected_session.get("summary", "")

    # === 3. 단답식 정답 추출 ===
    kw_model = KeyBERT(model="jhgan/ko-sroberta-multitask")
    keybert_keywords = [
        kw for kw, _ in kw_model.extract_keywords(
            summary,
            keyphrase_ngram_range=(1, 2),
            stop_words=None,
            top_n=30
        )
    ]
    kw_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    kw_prompt = ChatPromptTemplate.from_template("""
    당신은 뉴스 요약문에서 핵심 개념을 추출하는 **언어 정제 전문가**입니다.

    아래의 [요약문]과 [KeyBERT 후보]를 참고하여,
    문맥상 가장 중요한 **순수 명사 또는 복합 명사(2~3어절)** 7개를 최종 선별하세요.

    [정제 규칙 — 반드시 모두 지켜야 합니다]

    1. **요약문 안에 실제 등장한 명사**만 사용합니다.  
    - 새로운 단어, 의미 확장, 해석어를 만들어서는 안 됩니다.

    2. **명사(Noun)** 형태만 허용하며, 아래와 같은 어미나 품사는 모두 제외합니다:
    - 조사: 은, 는, 이, 가, 을, 를, 에, 에서, 으로, 와, 과, 등
    - 동사형 어미: 하다, 되다, 했다, 되고, 하는, 된, 되며 등
    - 형용사형 어미: 한, 된, 다양한, 큰, 작은, 새로운 등
    - 부사·접속사: 그러나, 또한, 특히, 즉, 따라서, 먼저 등
    - 추상어: 문제, 상황, 결과, 영향, 필요성, 변화, 관계, 과정, 요인, 측면 등
    - **날짜·시간(예: 2025년, 9월, 상반기 등)**, **수치·단위(예: %, 억, 개, 건, 회, 차 등)** 금지

    3. **2~3어절 복합 명사**는 허용합니다.  
    - 예: “전략적 동반자 관계”, “용산 대통령실”, “에너지 전환 정책”

    4. **아래 5개 카테고리 중 최소 1개 이상씩 포함되도록 7개를 고르세요.**
    - 인물명 (예: 대통령, 대표, 위원장 등 실제 인물 명칭)
    - 기관명 (예: 정부, 위원회, 기업, 공사, 단체 등)
    - 정책명 또는 제도명 (예: 정책, 계획, 전략, 제도, 협약 등)
    - 사건명 또는 활동명 (예: 정상회담, 발표, 구축, 수주, 개혁 등)
    - 핵심 개념 (예: 생성형 AI, 에너지 전환, 디지털 전환, 보안 체계 등)
        5. 중복·유사 표현은 하나로 통합합니다.  
        - 예: “AI 플랫폼”과 “생성형 AI 플랫폼”은 맥락상 하나로 간주
                                                 
    [정답 예시] :
    - “공정거래위원회는” → “공정거래위원회”
    - “과징금을 부과하기로” → “과징금”
    - “대학에서” → “대학”
    - “입시를 관리하다” → “입시”

    비허용 예시: “무역을”, “수시 모집에서”, “정책을 위한 계획”, “핵잠수함을 도입하는 것”, “경제적인 문제”
    허용 예시: “무역정책”, “핵잠수함 도입”, “가맹점 계약”, “학교폭력 기록”

    [요약문]
    {summary}

    [KeyBERT 후보]
    {keybert_keywords}
    """)

    kw_chain = kw_prompt | kw_llm
    kw_res = kw_chain.invoke({"summary": summary, "keybert_keywords": keybert_keywords})

    keywords_raw = kw_res.content.strip().replace("```", "").replace("json", "")
    keywords = [kw.strip() for kw in keywords_raw.split(",") if kw.strip()]

    logger.info(f"[LLM 키워드 정제 완료]")

    # === 4. 모델 설정 ===
    llm_i = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_e1 = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    llm_e2 = ChatOpenAI(model="gpt-4o", temperature=0.3)

    # === 5. YAML 프롬프트 로드 (I / E-1 / E-2 분리) ===
    def load_prompt_yaml(filename: str):
        """prompt 폴더 내 YAML 템플릿 로드"""
        prompt_dir = BASE_DIR / "src" / "quiz" / "prompt"
        path = prompt_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if "_type" not in data or "template" not in data:
            raise ValueError(f"잘못된 YAML 구조: {filename}")
        return ChatPromptTemplate.from_template(data["template"])

    # --- 프롬프트 개별 로드 ---
    prompt_i  = load_prompt_yaml("short_i.yaml")     # I단계 (기본 단답형)
    prompt_e1 = load_prompt_yaml("short_e_1.yaml")   # E-1단계 (상황절+의미절)
    prompt_e2 = load_prompt_yaml("short_e_2.yaml")   # E-2단계 (심화 리라이트형)

    # --- LLM 체인 구성 ---
    chain_i  = prompt_i  | llm_i
    chain_e1 = prompt_e1 | llm_e1
    chain_e2 = prompt_e2 | llm_e2

    # --- 공통 JSON 파싱 함수 ---
    def parse_json_output(res):
        """LLM 출력(JSON 문자열 또는 코드블록 포함)을 안전하게 파싱"""
        text = res.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("JSON 파싱 오류 발생, 빈 리스트 반환")
            return []

    # === 7. 문제 생성 ===
    try:
        # --- (1) I단계 ---
        i_res = chain_i.invoke({
            "summary": summary,
            "keywords": keywords
        })
        all_quiz_i = parse_json_output(i_res)
        logger.info(f"[I단계 생성 완료] {len(all_quiz_i)}문항")

        # === I단계 5문항 제한 ===
        i_quiz = all_quiz_i[:5]  # 상위 5개만 저장
        logger.info(f"[I단계 최종 선택] {len(i_quiz)}문항")

        # --- (2) E-1단계 ---
        e1_res = chain_e1.invoke({
            "summary": summary,
            "keywords": keywords
        })
        all_quiz_e1 = parse_json_output(e1_res)
        logger.info(f"[E-1단계 생성 완료] {len(all_quiz_e1)}문항")

        # --- (3) E-2단계 (I 심화형) ---
        e2_res = chain_e2.invoke({
            "i_quiz": json.dumps(all_quiz_i, ensure_ascii=False)
        })
        all_quiz_e2 = parse_json_output(e2_res)
        logger.info(f"[E-2단계 생성 완료] {len(all_quiz_e2)}문항")

        # --- (4) 병합 규칙 ---
        needed = max(0, 5 - len(all_quiz_e1))
        all_quiz_e = (all_quiz_e1 + all_quiz_e2[:needed])[:5]
        random.shuffle(all_quiz_e)
        e_quiz= all_quiz_e
        logger.info(f"[E단계 통합 완료] 최종 {len(all_quiz_e)}문항")

    except Exception as e:
        logger.error(f"[ERROR] {topic} 코스 {course_id} 세션 {session_id} 퀴즈 생성 중 오류: {e}")
        return

    # === 9. 저장 ===
    SAVE_DIR = BASE_DIR / "data" / "quiz"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")

    results = [
        {"contentType": "SHORT_ANSWER", "level": "I", "sourceUrl": sourceUrl, "contents": i_quiz},
        {"contentType": "SHORT_ANSWER", "level": "E", "sourceUrl": sourceUrl, "contents": e_quiz},
    ]

    for r in results:
        level = r["level"]
        filename = f"{topic}_{course_id}_{session_id}_SHORT_ANSWER_{level}_{today}.json"
        path = SAVE_DIR / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump([r], f, ensure_ascii=False, indent=2)
        logger.info(f"[저장 완료] {path.name}")

    logger.info(f"[완료] {topic} 코스 {course_id} 세션 {session_id} SHORT_ANSWER 생성 완료\n")


# === 실행 ===
if __name__ == "__main__":
    generate_short_quiz()