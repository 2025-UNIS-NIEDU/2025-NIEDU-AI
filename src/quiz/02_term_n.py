import os, re, json, requests
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from select_session import select_session

# === 1️. 환경 변수 로드 ===
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH, override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY")
GOOGLE_CSE_CX_DICT = os.getenv("GOOGLE_CSE_CX_DICT")
client = OpenAI(api_key=OPENAI_API_KEY)

# === 2. 세션 선택 ===
selected_session = select_session()
topic = selected_session["topic"]
course_id = selected_session.get("courseId")
session_id = selected_session.get("sessionId")
summary = selected_session["summary"]

print(f"\n선택된 토픽: {topic}")
print(f"courseId: {course_id}")
print(f"sessionId: {session_id}")
print(f"제목: {selected_session.get('headline', '')}\n")

# === 3️. 섹션별 전문용어 추출 템플릿 ===
PROMPT_TEMPLATES = {
    "politics": "정치·법률 관련 제도, 법률, 정책, 절차, 기관명 등 4개 추출",
    "economy": "경제·금융 관련 제도, 지표, 정책, 용어, 금융상품 등 4개 추출",
    "society": "사회·교육·복지 관련 제도, 정책, 사회현상, 제도명 등 4개 추출",
    "world": "국제정치·외교·안보 관련 제도, 협정, 기구, 정책, 용어 등 4개 추출",
    "tech": "과학·기술 관련 기술명, 개념, 시스템, 연구용어 등 4개 추출",
}

prompt = f"""
당신은 뉴스 요약문에서 핵심 전문용어를 추출하고 표제어 형태로 정제하는 전문가입니다.
아래 뉴스의 주제는 **{topic}**이며, 다음 기준에 따라 용어를 4개만 추출하세요.

🎯 추출 기준:
- {PROMPT_TEMPLATES[topic]}
- 실제 뉴스 요약문에 등장한 단어만 사용
- 인명, 기관명, 지명, 기업명 제외
- 상업적 목적과 연관된 단어 제외
- 불필요한 수식어나 조사 제거 (예: ~추진, ~계획, ~논의 등)
- 각 용어는 1~3단어의 명사 형태
- 쉼표로 구분하여 출력 (예: 탄소중립, 전력시장, 재생에너지, 전기요금제)
- 추가 설명이나 문장은 출력하지 말고, 쉼표로 구분된 단어만 출력하세요.

[뉴스 요약문]
{summary}
"""

res = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)

terms = [t.strip() for t in res.choices[0].message.content.strip().split(",") if t.strip()]
terms = terms[:4]  # 혹시 5개 이상 나오면 앞의 4개만 사용

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
        return term, None

    data = res.json()
    items = data.get("items", [])
    if not items:
        return term, None

    # === snippet 가장 긴 항목 선택 ===
    best_item = max(
        items,
        key=lambda x: len(x.get("snippet", "")) if x.get("snippet") else 0
    )

    title = best_item.get("title", "")
    snippet = best_item.get("snippet", "")

    # === snippet 정제 ===
    snippet = snippet.strip() if snippet else None

    return term, snippet

# === 5.5 용어 정의 완성 ===
def complete_snippet(term, snippet):
    prompt = f"""
다음 문장은 '{term}'의 정의 일부로 보입니다.

1️. 먼저, 주어진 문장을 **한글로만** 구성된 **완전한 문장**으로 자연스럽게 이어서 완성하세요.
2️. 만약 원문이 지나치게 끊기거나, 이은 문장이 문법적으로나 의미상으로 말이 되지 않는다면
   — 그때는 '{term}'의 정의를 **새롭게 한 문장으로 재작성**하세요.
3️.불필요한 기호(..., ·, :, -, “”, 등)는 모두 제거하세요.

규칙:
- 반드시 한 문장으로 작성
- 문체는 사전식 정의체 (“~이다.”, “~를 의미한다.”, “~을 말한다.” 등)
- 주어진 내용이 자연스럽게 끝나면 새로운 내용은 추가하지 말 것
- 문장이 끝날 때 반드시 마침표로 종료
- 영어나 숫자는 그대로 두되, 불필요한 영어 설명은 제거

[입력 문장]
{snippet}
"""
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    completed = res.choices[0].message.content.strip()
    return completed

# === 6️. 예시 문장 + 비유 생성 ===
def build_examples(term, news_text):
    prompt_1 = f"""
    다음 뉴스 내용을 참고하여 "{term}"이(가) 포함된 자연스럽고 완전한 한 문장을 만들어주세요.
    - 실제 뉴스 문체(보도체)를 유지하세요.
    - 문장이 중간에서 끊기지 않도록 완전하게 작성하세요.
    - 문체나 어휘가 다른 용어의 문장과 **중복되지 않도록** 다양하게 표현
    - 100자 이내로 1문장만 출력하세요.

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
    "{term}"을(를) 일상적 비유로 120자 이내로 간단히 설명하세요.
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

# === 7️. 용어별 카드 생성 ===
results = []
for term in terms:
    term, snippet = fetch_definition(term)
    completed = complete_snippet(term, snippet) if snippet else None
    example, analogy = build_examples(term, summary)
    results.append({
        "text": term,
        "definition": completed or snippet,
        "exampleSentence": example,
        "analogy": analogy,
    })

# === 8️. NIEdu 포맷 ===
term_card = {
    "courseId": course_id,
    "sessionId": session_id,
    "topic" : topic,
    "contentType": "term",
    "level": "n",
    "items": [
        {"question": None, "answers": results}
    ]
}

# === 9️. 결과 출력 및 저장 ===
print("\n=== 뉴스 요약문 ===")
print(summary)
print("\n=== 변환된 NIEdu 용어 카드 ===")
print(json.dumps(term_card, ensure_ascii=False, indent=2))

QUIZ_DIR = BASE_DIR / "data" / "quiz"
QUIZ_DIR.mkdir(parents=True, exist_ok=True)
today = datetime.now().strftime("%Y-%m-%d")
save_path = QUIZ_DIR / f"{topic}_term_n_{today}.json"
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(term_card, f, ensure_ascii=False, indent=2)

print(f"\n 저장 완료: {save_path}")
print(f"({len(results)}개 용어 추출됨)")