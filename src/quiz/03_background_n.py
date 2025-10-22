import os, re, json, requests
from openai import OpenAI
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from select_session import select_session   

# === 1. 세션 불러오기 ===
selected_session = select_session()
topic = selected_session["topic"]
course_id = selected_session.get("courseId")
session_id = selected_session.get("sessionId")
summary = selected_session["summary"]

print(f"\n선택된 토픽: {topic}")
print(f"courseId: {course_id}")
print(f"sessionId: {session_id}")
print(f"제목: {selected_session.get('headline', '')}\n")

# === 2️. 환경 변수 로드 ===
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH, override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY")
GOOGLE_NEWS_ID = os.getenv("GOOGLE_CSE_CX_NEWS")
GOOGLE_GOV_ID = os.getenv("GOOGLE_CSE_CX_GOV")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# === 3️. CSE 검색 함수 ===
def get_cse_snippets(query, cx_id, n=10):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"q": query, "key": GOOGLE_API_KEY, "cx": cx_id, "num": n}
    res = requests.get(url, params=params)
    if res.status_code != 200:
        print(f"❌ CSE API 오류: {res.status_code}")
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

print(f"스니펫 수집 완료: 뉴스 {len(news_snippets)} + 정부 {len(gov_snippets)} = 총 {len(snippets)}개")

# === 5️. 스니펫을 하나의 컨텍스트로 병합 ===
context = "\n\n".join(
    [f"[{i+1}] {s['title']}\n{s['snippet']}\nURL: {s['link']}" for i, s in enumerate(snippets)]
)

# === 6️. LLM 프롬프트 구성 ===
prompt_background_n = f"""
당신은 언론 분석가입니다.
아래 기사 요약문과 CSE 검색 스니펫을 참고하여,
사건의 전후 맥락과 의미를 다섯 가지 항목(이슈명, 원인, 상황, 결과, 영향)으로 요약하세요.

모든 항목은 반드시 존재해야 하며, 각 항목별로 'text'(설명)과 'sourceUrl'(출처 URL)을 포함하세요.
출력은 **JSON 형식**으로만 작성하고, 불필요한 설명은 하지 마세요.

⚙️ 항목별 규칙:
1. **이슈명** (기사 주제·핵심 키워드): 40자 이내, 한 줄 요약
2. **원인** (사건 발생 배경·원인 요인): 최소 100자 이상, 120자 이내 구체적으로
3. **상황(타임라인)** (시점별 전개·주요 행위자): 최소 100자 이상, 120자 이내
4. **결과** (단기적 결과·즉각적 변화): 최소 80자 이상, 100자 이내
5. **영향** (장기적 사회·경제·정치적 함의): 최소 80자 이상, 100자 이내

⚙️ 작성 규칙:
- text는 지정된 글자 수 범위 내에서 충분히 구체적이고 자연스러운 문장으로 작성
- 각 항목은 중복 없이 구체적으로
- 'sourceUrl'은 snippet 중 관련성 높은 기사 링크를 사용
- **JSON만 출력**, 추가 설명 금지

출력 예시:
{{
  "level": "N",
  "currentAffair": [
    {{"label": "이슈명", "text": "...", "sourceUrl": "..." }},
    {{"label": "원인", "text": "...", "sourceUrl": "..." }},
    {{"label": "상황(타임라인)", "text": "...", "sourceUrl": "..." }},
    {{"label": "결과", "text": "...", "sourceUrl": "..." }},
    {{"label": "영향", "text": "...", "sourceUrl": "..." }}
  ]
}}

---
[기사 요약문]
{selected_session.get("summary")}

[관련 검색 스니펫 (총 {len(snippets)}개)]
{context}
"""

# === 7️. LLM 호출 ===
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt_background_n}],
    temperature=0.4
)

raw_output = resp.choices[0].message.content.strip()

# === 8. JSON 파싱 및 NIEdu 포맷 감싸기 ===
try:
    parsed = json.loads(re.sub(r"```json|```", "", raw_output))
except json.JSONDecodeError:
    print(" JSON 파싱 실패 — 원문 그대로 저장합니다.")
    parsed = {"level": "N", "currentAffair": []}

# LLM이 "currentAffair" 구조로 줬을 경우 보정
if "currentAffair" in parsed:
    answers = []
    for b in parsed["currentAffair"]:
        answers.append({
            "label": b.get("label"),
            "text": b.get("text"),
            "sourceUrl": b.get("sourceUrl")
        })
else:
    # 혹시 배열 형태로 바로 반환된 경우
    answers = parsed if isinstance(parsed, list) else []

background_card = {
    "courseId": course_id,
    "sessionId": session_id,
    "topic" : topic,
    "contentType": "background",
    "level": "n",
    "items": [
        {
            "question": None,
            "answers": answers
        }
    ]
}

# === 9️. 결과 출력 및 저장 ===
print("\n=== 뉴스 요약문 ===")
print(summary)
print("\n=== 변환된 NIEdu 포맷 ===")
print(json.dumps(background_card, ensure_ascii=False, indent=2))

# === 10. 저장 ===
SAVE_DIR = BASE_DIR / "data" / "quiz"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
today = datetime.now().strftime("%Y-%m-%d")
file_path = SAVE_DIR / f"{selected_session['topic']}_background_n_{today}.json"

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(background_card, f, ensure_ascii=False, indent=2)

print(f"\n 저장 완료: {file_path}")