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

    logger.info(f"[{topic}] 세션 {session_id} SUMMARY_READING 시작 — 제목: {headline}")

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
    너는 뉴스의 핵심을 요약하는 분석가이다.

    다음 뉴스를 읽고,
    1. 사건의 **주체(Actor)** 를 1개 선택하고,
    2. 사건의 **핵심 개념(Object)** 을 1개 선택하라.
    핵심 개념은 **하나의 명사** 또는 **2~3어절 명사구** 형태로 제시하라.

    출력 형식(JSON):
    {{
    "keywords": [
        {{"word": "단어1"}},
        {{"word": "단어2"}}
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
    bert_keywords = kw_model.extract_keywords(refined_summary, keyphrase_ngram_range=(1, 2), top_n=15)

    # SentenceTransformer 로 anchor 유사도 필터링
    embed_model = SentenceTransformer("jhgan/ko-sroberta-multitask")

    def is_related(word, anchors, threshold=0.55):
        word_emb = embed_model.encode(word, convert_to_tensor=True)
        sims = [util.cos_sim(word_emb, embed_model.encode(a, convert_to_tensor=True)).item() for a in anchors]
        return max(sims) >= threshold

    related_candidates = [k for k, v in bert_keywords if is_related(k, anchor_words)]
    logger.info(f"[{topic}] 정답과 관련된 명사 후보: {related_candidates}")

    # === 3. LLM 혼동 가능성 평가 ===
    prompt_confuse = f"""
    너는 뉴스 학습용 문제를 설계하는 전문가이다.

    다음 뉴스 요약문과 정답 단어를 참고하여,
    정답과 주제적으로 밀접하지만 의미가 완전히 일치하지 않는 단어들을 오답 후보로 제시하라.

    단, 모든 출력 단어는 **순수한 명사 또는 명사구 형태**여야 하며,
    조사(은, 는, 이, 가, 을, 를 등)나 어미(하다, 되다 등)를 절대 포함하지 않는다.

    예시:
    ❌ "무역을", "경제적인"  → X  
    ✅ "무역", "경제", "교역 정책"  → O

    [오답 후보]
    {bert_keywords}

    [입력]
    뉴스 요약문:
    {refined_summary}

    정답 단어: {answers}

    [출력 형식]
    {{
    "ranked": [
        {{"word": "<단어>", "score": 0.xx, "reason": "<헷갈릴 이유>"}}
    ]
    }}
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

    # === 5. 유의어/중복 필터링 ===
    def is_semantically_similar(word, answers, threshold=0.8):
        word_emb = embed_model.encode(word, convert_to_tensor=True)
        for a in answers:
            ans_emb = embed_model.encode(a, convert_to_tensor=True)
            sim = util.cos_sim(word_emb, ans_emb).item()
            if sim >= threshold:
                return True
        return False

    def too_similar(word, answers):
        return any(a in word or word in a for a in answers)

    filtered_combined = {
        w: s for w, s in combined.items()
        if not too_similar(w, answers)
        and not is_semantically_similar(w, answers, threshold=0.8)
    }

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

    logger.info(f"[{topic}] 세션 {session_id} SUMMARY_READING 퀴즈 생성 완료")

# === 실행 ===
if __name__ == "__main__":
    generate_summary_reading_quiz()