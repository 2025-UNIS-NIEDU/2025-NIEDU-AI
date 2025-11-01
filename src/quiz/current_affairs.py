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
   - 예: “법제처장 ‘무죄 발언’ 논란, 공직 중립성 논의 확산”

2. **cause (사건의 배경 및 촉발 요인)**  
   - 100~130자.  
   - 사건이 발생한 제도적·정치적·사회적 배경을 구체적으로 기술.
   - 단순한 계기가 아니라, ‘왜 이 사건이 문제로 부상했는가’를 설명.  
   - 인물의 행위·제도적 맥락·정치적 배경을 구체적으로 명시.

3. **circumstance (전개 과정 및 핵심 반응)**  
   - 100~130자.  
   - 시점별 주요 행위자(정부·정당·기관 등)의 입장과 반응을 비교 중심으로 기술.  
   - 단순한 반응 나열이 아니라, **행위자 간의 상호작용 구조**가 드러나야 한다.
   - 만약 요약문에 행위자·이유·배경 등 핵심 정보가 누락되어 공백이 생긴다면,
     기사에 드러난 맥락을 활용해 논리적으로 자연스러운 연결 문장으로 복원하라.

4. **result (즉각적 결과·정치적 파장)**  
   - 80~110자.  
   - 논란 이후의 실질적 변화, 여론·정치권 반응·조치 등을 중심으로 기술.
   - 여론·정책 변화·조치 등 구체적 후속 반응을 중심으로 기술.

5. **effect (장기적 영향·사회적 함의)**  
   - 80~110자.  
   - 제도적 신뢰, 정치 윤리, 사회 구조 등 보다 근본적 차원의 시사점을 제시.
   - 단순한 영향이 아니라 “이 사건이 사회에 던진 질문”이나 “재발 방지·제도 개선의 필요성” 등 학습적 통찰로 마무리한다.

[작성 규칙]
- 각 항목은 **"text"**(설명)과 **"sourceUrl"**(출처 URL)을 모두 포함한다.  
- 모든 문장은 자연스러운 한국어 기사체로 작성한다.  
- 중복 표현 금지, 논리적 연결 유지.  
- **추측 금지. 기사 또는 스니펫에 언급된 근거만 활용.**  
- 출력은 반드시 **JSON 형식**으로만 작성할 것. (추가 설명, 서두, 해설 금지)

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
    model="gpt-4o",
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