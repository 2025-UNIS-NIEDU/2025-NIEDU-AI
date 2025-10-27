import os, re, json, requests
from openai import OpenAI
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from select_session import select_session   

# === 1. 환경 변수 로드 ===
load_dotenv(override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY")
GOOGLE_NEWS_ID = os.getenv("GOOGLE_CSE_CX_NEWS")
GOOGLE_GOV_ID = os.getenv("GOOGLE_CSE_CX_GOV")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# === 2. 세션 불러오기 ===
selected_session = select_session()
topic = selected_session["topic"]
course_id = selected_session["courseId"]            
session_id = selected_session.get("sessionId")
headline = selected_session.get("headline", "")
summary = selected_session.get("summary", "")

print(f"\n선택된 코스: {course_id}")
print(f"sessionId: {session_id}")
print(f"제목: {headline}\n")

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

[항목별 규칙]
1. **issue** (기사 주제·핵심 키워드): 40자 이내, 한 줄 요약
2. **cause** (사건 발생 배경·원인 요인): 최소 100자 이상, 120자 이내 구체적으로
3. **circumstance** (시점별 전개·주요 행위자): 최소 100자 이상, 120자 이내
4. **result** (단기적 결과·즉각적 변화): 최소 80자 이상, 100자 이내
5. **effect** (장기적 사회·경제·정치적 함의): 최소 80자 이상, 100자 이내

[작성 규칙]
1. text는 지정된 글자 수 범위 내에서 충분히 구체적이고 자연스러운 문장으로 작성
2. 각 항목은 중복 없이 구체적으로
3. **JSON만 출력**, 추가 설명 금지

출력 예시:
{{
  "CURRENT_AFFAIRS": {{
    "issue": "...",
    "cause": "...",
    "circumstance": "...",
    "result": "...",
    "effect": "..."
  }}
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

# === 8. JSON 파싱 ===
try:
    cleaned = re.sub(r"```json|```", "", raw_output)
    parsed = json.loads(cleaned)
except json.JSONDecodeError:
    print("JSON 파싱 실패 — 원문 그대로 저장합니다.")
    parsed = {
        "CURRENT_AFFAIRS": {
            "issue": "",
            "cause": "",
            "circumstance": "",
            "result": "",
            "effect": ""
        }
    }

# === 구조 검증 및 보정 ===
affairs = parsed.get("CURRENT_AFFAIRS", {})
answers = {
    "issue": affairs.get("issue", ""),
    "cause": affairs.get("cause", ""),
    "circumstance": affairs.get("circumstance", ""),
    "result": affairs.get("result", ""),
    "effect": affairs.get("effect", "")
}

# === 최종 출력 (NIEdu 통합 구조)
wrapped_output = {
    "contentType": "CURRENT_AFFAIRS",
    "level": "N",
    "contents": [answers],
}

# 저장 또는 출력
BASE_DIR = Path(__file__).resolve().parents[2]
SAVE_DIR = BASE_DIR / "data" / "quiz"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
today = datetime.now().strftime("%Y-%m-%d")

file_path = SAVE_DIR / f"{topic}_{course_id}_{session_id}_CURRENT_AFFAIRS_N_{today}.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(wrapped_output, f, ensure_ascii=False, indent=2)

print(f"CURRENT_AFFAIRS 저장 완료: {file_path}")