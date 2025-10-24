import os, json, re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from kiwipiepy import Kiwi
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
from select_session import select_session

# === 환경 변수 로드 ===
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# === 세션 선택 ===
selected_session = select_session()
topic = selected_session["topic"]
coourse_id = selected_session["courseId"]            
session_id = selected_session.get("sessionId")
headline = selected_session.get("headline", "")
summary = selected_session.get("summary", "")

print(f"\n선택된 코스: {coourse_id}")
print(f"sessionId: {session_id}")
print(f"제목: {headline}\n")

# === KIWI 명사 추출 ===
kiwi = Kiwi()
nouns = [t.form for t in kiwi.tokenize(summary) if t.tag.startswith("N")]
nouns = list(dict.fromkeys([n for n in nouns if len(n) > 1]))

# === 핵심 정답 추출 ===
prompt_answer = f"""
너는 뉴스의 핵심을 요약하는 분석가이다.
아래 뉴스 을 읽고,
1) 사건의 **주체(Actor)**,
2) 사건의 **핵심 개념(Object)**,
이 두 가지를 명확하게 선택하라.

출력 형식(JSON):
{{
  "answers": [
    {{"word": "<단어>", "reason": "<이유>"}},
    {{"word": "<단어>", "reason": "<이유>"}}
  ]
}}

명사 목록:
{nouns}

뉴스 요약문:
{summary}
"""
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt_answer}],
    temperature=0
)
answers_json = json.loads(re.sub(r"```json|```", "", resp.choices[0].message.content.strip()))
answers = [a["word"] for a in answers_json["answers"]]
print(f"중심 키워드(정답): {answers}")

# === KeyBERT 후보 점수화 ===
kw_model = KeyBERT(model="jhgan/ko-sroberta-multitask")
bert_keywords = kw_model.extract_keywords(summary, keyphrase_ngram_range=(1, 1), top_n=30)
bert_ranked = {k: v for k, v in bert_keywords if k in nouns and k not in answers}

# === 혼동 가능성 평가 ===
prompt_confuse = f"""
너는 뉴스 문맥을 이해하고 각 단어가 정답 단어들과 얼마나 헷갈릴 만한지를 평가하는 분석가이다.
정답 단어: {answers}

각 단어의 혼동 가능성(confusability)을 0~1로 평가하라.
단, **가장 헷갈리는 단어는 1.0, 전혀 관련 없는 단어는 0.0**으로 스케일링하라.

출력 형식(JSON):
{{
  "ranked": [
    {{"word": "<단어>", "score": 0.xx, "reason": "<이유>"}}, ...
  ]
}}

명사 목록:
{nouns}
뉴스 요약문:
{summary}
"""
resp2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt_confuse}],
    temperature=0
)
llm_ranked_data = json.loads(re.sub(r"```json|```", "", resp2.choices[0].message.content.strip()))
llm_ranked = {item["word"]: item["score"] for item in llm_ranked_data["ranked"] if item["word"] in nouns}

# === 점수 결합 ===
alpha = 0.7
combined = {
    word: round(alpha * llm_ranked.get(word, 0) + (1 - alpha) * bert_ranked.get(word, 0), 3)
    for word in set(bert_ranked) | set(llm_ranked)
}

# === 정규화 ===
min_s, max_s = min(combined.values()), max(combined.values())
for w in combined:
    combined[w] = round((combined[w] - min_s) / (max_s - min_s + 1e-6), 3)

# === 유의어 필터링 ===
embed_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
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

# === 난이도별 분류 ===
distractors = sorted(filtered_combined.items(), key=lambda x: x[1], reverse=True)[:9]
step = max(1, len(distractors) // 3)
e_level = distractors[:step]
i_level = distractors[step:step * 2]
n_level = distractors[step * 2:]

# === JSON 생성 ===
def make_keyword_block(level, correct_list, distractors):
    correct_two = correct_list[:2] if len(correct_list) >= 2 else correct_list
    answers_block = [{"text": c, "isCorrect": True, "explanation": None} for c in correct_two]
    answers_block += [{"text": d[0], "isCorrect": False, "explanation": None} for d in distractors[:3]]
    return {
        "topic" : topic,
        "coourseId": coourse_id,
        "sessionId": session_id,
        "contentType": "keyword",
        "level": level,
        "items": [{"question": None, "answers": answers_block}]
    }

correct_list = answers[:2] if len(answers) >= 2 else answers
final_json = [
    make_keyword_block("n", correct_list, n_level),
    make_keyword_block("i", correct_list, i_level),
    make_keyword_block("e", correct_list, e_level)
]

# === 9️. 결과 출력 및 저장 ===
print("\n=== 뉴스 요약문 ===")
print(summary)
print("\n=== 변환된 NIEdu 포맷 ===")
print(json.dumps(final_json, ensure_ascii=False, indent=2))

BASE_DIR = Path(__file__).resolve().parents[2]
QUIZ_DIR = BASE_DIR / "data" / "quiz"
QUIZ_DIR.mkdir(parents=True, exist_ok=True)
today = datetime.now().strftime("%Y-%m-%d")
file_path = QUIZ_DIR / f"{topic}_{coourse_id}_{session_id}_keyword_nie_{today}.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(final_json, f, ensure_ascii=False, indent=2)

print(f"\n저장 완료: {file_path.resolve()}")