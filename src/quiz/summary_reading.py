import os, json, re, logging
from datetime import datetime
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
from openai import OpenAI
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
from quiz.select_session import select_session

logger = logging.getLogger(__name__)

def generate_summary_reading_quiz(selected_session=None):
    # === 환경 변수 로드 ===
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
    너는 '뉴스 문해력 학습용 교재에 수록될 요약문'을 교정하는 전문 편집자이다.  
    문체는 보도체가 아닌 분석적 서술체로 유지한다.  

    [교정 기준]
    1. 원문에 존재하지 않는 정보나 추측을 추가하지 않는다.  
    2. 인물, 기관, 수치 등 사실 관계는 원문과 일치하도록 유지한다.  
    3. 모든 문장은 과거 시제로 통일한다.  
    4. 문장은 중립적이고 사실 서술형으로 작성한다.  
    - 감정적, 비유적, 추측성 어휘(예: '논란', '비판적', '충격')는 삭제한다.  
    5. 각 문장은 25자 내외로 간결하게, 한 문단이 한 호흡으로 자연스럽게 이어지게 한다.
    6. 인물·기관은 처음 등장할 때 전체 명칭으로, 이후 약칭 사용 가능하다. 부족한 정보가 있다면 명칭 간단히 덧붙인다. 
    7. 대명사(그, 이, 해당 등)와 지시어(이날, 이에 대해 등)는 구체적인 명사로 바꾼다.  
    8. 날짜·시간 표현은 가능한 한 명시적 표현으로 바꾼다. (‘오늘’ → ‘29일’)  
    9. 전체는 사건의 발생→발언→결과 순으로 자연스럽게 배열한다.  

    [주의]
    1. 단, 문장을 매끄럽게 만들기 위해 어순 조정은 허용된다.  
    2. 띄어 쓰기에 유의한다.
    3. 최종 출력은 아래 JSON 형식으로만 반환하라.

    {{
    "summary": "<교정된 요약문 (약 200~250자)>"
    }}

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

    logger.info(f"[{topic}] 1차 요약문 정제 완료")

    prompt_refine = f"""
    + 너는 '뉴스 문해력 학습용 교재에 수록될 요약문'을 교정하는 전문 편집자이다.  
    아래 [입력 요약문]을 **다시 작성하라.**  

    [지침]
    1. 사건의 전개는 원인 → 전개 → 결과 순으로 자연스럽게 배열한다.
    2. 누가, 언제, 무엇을, 왜 했는지가 명확히 드러나야 한다.
    3. 문체는 중립적이고 사실 중심의 보도체로 유지하되, 어미는 자연스럽게 조정한다.
    4. 세부 정보가 불확실할 때는 추정하지 않는다. 대신, 문맥이 단절되지 않도록 기사 속에서 확인 가능한 근거나 발언, 상황 묘사를 활용해 자연스럽게 연결한다.
    만약 원인이 명확히 드러나지 않는다면, 초점을 인물의 발언 배경이나 정책적 의미로 이동시켜 서술한다.5. 문장 간 흐름이 매끄럽도록 연결어를 조정하되, 논리를 억지로 이어 붙이지 않는다.
    6. 전체 문단은 200~250자 내외로 구성한다.

    [주의]
    1. 새로운 사실이나 세부 정보를 창작하지 말라.  
    2. 띄어 쓰기에 유의한다. 
    3. 출력은 반드시 아래 JSON 형식으로만 반환하라.

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

    logger.info(f"[{topic}] 최종 요약문 정제 완료")

    # === 핵심 정답 추출 ===
    prompt_answer = f"""
    너는 뉴스의 핵심 구조를 분석하는 정보 추출 전문가이다.

    다음 요약문을 읽고, 사건의 중심 요소 2가지를 식별하라.

    [추출 대상]
    1. **주체(Actor)**: 사건의 중심이 되는 인물, 기관, 또는 단체.  
    - 행동을 주도하거나 발언·결정을 내린 주체를 한 개만 선택하라.  
    - 예시: '이재명 대통령', '공정거래위원회', '국방부', '오바마 전 대통령'  
    - 복수의 기관·인물이 등장하더라도, 주된 역할을 수행한 하나만 선택한다.

    2. **핵심 개념(Object)**: 사건의 본질적 주제나 핵심 사안.  
    - 주체가 행한 행동이나 정책, 또는 논의의 중심 개념을 한 개만 선택하라.  
    - 예시: '과징금 부과', '전략적 동반자 관계', '학교폭력', '입시 제도'  
    - 단, ‘문장 일부’(예: ‘과징금을 부과하기로’)처럼 동사형이나 조사가 포함된 형태는 금지한다.  
    - 반드시 **명사 또는 2~3어절 복합 명사** 형태로 제시하라.  
    - 수치, 날짜, 단위(예: 2023년, 6억 원 등)는 포함하지 않는다.

    [출력 규칙]
    - 모든 단어는 실제 요약문에 등장해야 한다.  
    - 조사가 포함된 경우(은, 는, 이, 가, 을, 를, 에, 으로 등)는 제거하고 명사형만 남긴다.  
    - 결과는 JSON 형식으로만 출력한다.

    [출력 형식 예시]
    {{
    "keywords": [
        {{"word": "공정거래위원회"}},    ← 주체(Actor)
        {{"word": "과징금"}}             ← 핵심 개념(Object)
    ]
    }}

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
    logger.info(f"[{topic}] 중심 키워드(정답): {answers}")

    # === 2. KeyBERT로 관련 단어 필터링 ===
    anchor_words = answers
    kw_model = KeyBERT(model="jhgan/ko-sroberta-multitask")
    bert_keywords = kw_model.extract_keywords(refined_summary, keyphrase_ngram_range=(1, 2), top_n=30)

    # SentenceTransformer 로 anchor 유사도 필터링
    embed_model = SentenceTransformer("jhgan/ko-sroberta-multitask")

    def is_related(word, anchors, threshold=0.5):
        word_emb = embed_model.encode(word, convert_to_tensor=True)
        sims = [util.cos_sim(word_emb, embed_model.encode(a, convert_to_tensor=True)).item() for a in anchors]
        return max(sims) >= threshold

    related_candidates = [k for k, v in bert_keywords if is_related(k, anchor_words)]
    logger.info(f"[{topic}] 정답과 관련된 명사 후보: {related_candidates}")

    # === 3. LLM 혼동 가능성 평가 ===
    prompt_confuse = f"""
    너는 뉴스 요약문에서 핵심 정보를 추출하는 전문가이다.

    다음 요약문과 정답 단어 2개(주체/핵심개념)를 기반으로,
    학생이 헷갈릴 수 있는 오답 후보를 제시하라.

    [정답 역할 구분]
    1. 첫 번째 단어 → 사건의 주체(Actor)
    2. 두 번째 단어 → 사건의 핵심 개념(Object)

    ---

    [오답 생성 원칙]

    (1) 주체(Actor) 관련 오답:
    - 실제 기사에 등장하지만 사건의 주체가 아닌 명사.
    - 기관명, 조직명, 직위, 집단명, 직책, 국가, 정당 등.
    - 예시: 정답이 ‘이재명 대통령’일 경우 → ‘대통령실’, ‘정부’, ‘국방부’, ‘야당’, ‘호주’ 등.

    (2) 핵심 개념(Object) 관련 오답:
    - 기사 맥락상 핵심 개념과 연관은 있으나 중심이 아닌 부차적 요소.
    - 정책명, 사건명, 협약, 제도, 수단, 결과물, 부연 설명어 등.
    - 예시: 정답이 ‘핵잠수함 도입’일 경우 → ‘합의’, ‘협정’, ‘자주국방’, ‘호주 선례’, ‘외교 실패’ 등.

    ---

    [주의사항]
    - 모든 단어는 조사나 어미가 없는 명사 또는 2~3어절 복합 명사여야 한다.
    - 정답의 동의어, 축약형, 포함 관계(예: ‘핵잠수함’ vs ‘핵잠’)는 금지한다.
    - 추상적·메타적 단어(‘문제’, ‘결과’, ‘상황’, ‘의미’, ‘이유’)는 제외한다.
    - 조사(은, 는, 이, 가, 을, 를, 에, 에서, 으로, 와, 과)나 
    동사형 어미(하다, 되다, 하기로, 하였다, 했다 등)가 포함된 경우 해당 부분을 제거하고 순수 명사만 남긴다.

    예시:
    - “공정거래위원회는” → “공정거래위원회”
    - “과징금을 부과하기로” → “과징금”
    - “대학에서” → “대학”
    - “입시를 관리하다” → “입시”

    비허용 예시: “무역을”, “수시 모집에서”, “정책을 위한 계획”, “핵잠수함을 도입하는 것”, “경제적인 문제”
    허용 예시: “무역정책”, “핵잠수함 도입”, “가맹점 계약”, “학교폭력 기록”

    ---

    [출력 형식]
    {{
    "ranked": [
        {{"word": "정부", "role": "actor", "score": 0.85, "reason": "기사에 등장하지만 주체가 아님"}},
        {{"word": "정책", "role": "object", "score": 0.82, "reason": "핵심 개념과 연관은 있으나 중심이 아님"}}
    ]
    }}

    [출력 조건]
    - 최소 14개 이상 제시할 것.
    - 주체형(actor) 오답과 개념형(object) 오답이 균형 있게 포함될 것.
    - 각 단어에 대해 학생이 왜 혼동할 수 있는지를 간단히 설명할 것.

    ---

    [입력 데이터]
    요약문:
    {refined_summary}

    정답 단어:
    {answers}

    KeyBERT 후보 단어:
    {related_candidates}
    """

    resp_confuse = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_confuse}],
        temperature=0
    )

    try:
        llm_ranked_data = json.loads(re.sub(r"```json|```", "", resp_confuse.choices[0].message.content.strip()))
    except json.JSONDecodeError as e:
        logger.error(f"[{topic}] JSON 파싱 오류(혼동어): {e}", exc_info=True)
        raise

    llm_ranked = {item["word"]: item["score"] for item in llm_ranked_data["ranked"]}
    logger.info(f"[{topic}] 혼동어 후보: {list(llm_ranked.keys())}")

    # === 4. 정규화 ===
    min_s, max_s = min(llm_ranked.values()), max(llm_ranked.values())
    combined = {
        w: round((s - min_s) / (max_s - min_s + 1e-6), 3)
        for w, s in llm_ranked.items()
    }

    # === 5. 유의어/중복 필터링 (완화 + KeyBERT 보충) ===
    def is_semantically_similar(word, answers, threshold=0.9):
        word_emb = embed_model.encode(word, convert_to_tensor=True)
        for a in answers:
            ans_emb = embed_model.encode(a, convert_to_tensor=True)
            sim = util.cos_sim(word_emb, ans_emb).item()
            if sim >= threshold:
                return True
        return False

    def too_similar(word, answers):
        # 완전히 동일한 단어만 제외
        return any(a == word for a in answers)

    filtered_combined = {
        w: s for w, s in combined.items()
        if not too_similar(w, answers)
        and not is_semantically_similar(w, answers, threshold=0.9)
    }

    # === 오답 후보 부족 시 KeyBERT 기반 보충 ===
    if len(filtered_combined) < 9:
        logger.warning(f"[{topic}] 오답 후보 부족 → KeyBERT 보충 시작 ({len(filtered_combined)}개)")
        for k, v in bert_keywords:
            k = k.strip()
            if k not in filtered_combined and k not in answers and len(k) > 1:
                filtered_combined[k] = 0.15  # 낮은 점수로 추가
            if len(filtered_combined) >= 9:
                break

    if len(filtered_combined) < 9:
        logger.warning(f"[{topic}] KeyBERT 보충 후에도 부족 ({len(filtered_combined)}개), 일부 중복 허용")
        for k, v in bert_keywords:
            if k not in filtered_combined:
                filtered_combined[k] = 0.1
            if len(filtered_combined) >= 9:
                break

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