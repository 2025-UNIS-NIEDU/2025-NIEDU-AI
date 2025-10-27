import os, json, random
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from select_session import select_session

# === 환경 변수 로드 ===
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH, override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === 세션 선택 ===
selected_session = select_session()
topic = selected_session["topic"]
course_id = selected_session["courseId"]            
session_id = selected_session.get("sessionId")
headline = selected_session.get("headline", "")
summary = selected_session.get("summary", "")
sourceUrl = selected_session.get("sourceUrl", "")

print(f"\n선택된 태그: {course_id}")
print(f"sessionId: {session_id}")
print(f"제목: {headline}\n")

# === 모델 설정 ===
llm_i = ChatOpenAI(model="gpt-4o", temperature=0.3)  # I단계
llm_e = ChatOpenAI(model="gpt-5") # E단계

# === I단계 문제 생성 ===
def generate_quiz_i(summary: str):
  prompt_i = f"""
당신은 뉴스 기반 학습 퀴즈 생성 AI입니다.
아래 뉴스 내용을 참고하여 단답식 문제를 10개 만들어주세요.

[문항 구성 기준]
1. 질문은 뉴스에서 다룬 **핵심 개념·정책·사건·제도명** 중심으로 작성합니다.
2. 인물명·지명·시간 관련 표현·수치 등은 절대 정답으로 사용하지 않습니다.
3. 각 문항은 서로 다른 사실 또는 관점 요소를 다룹니다.  

[정답 생성 규칙]
1. 정답은 반드시 명사 한 단어로 작성합니다.  
2. 모든 정답은 중복되지 않게 합니다.
3. 시간 표현, 평가·감정 표현, 추상적 행위명사, 대명사·지시사 등은 정답으로 사용하지 않습니다.  
4. 정답은 자동 채점이 가능한 단어여야 하며, 의미가 유사하거나 변형 가능한 단어는 모두 금지합니다.  
5. 정답은 뉴스에 명시된 **한 단어 명사**로, 사람, 조직, 제도, 정책, 사건, 혹은 명확한 사물 이름이어야 합니다.

[세부 규칙]
1. question은 최소 30자, 40자 이내, 한 문장, 의문형 문어체
2. 해설은 50자 이내, 사실적 근거 중심, 모든 해설 문장은 한 문장으로 작성하고, ‘~다.’로 끝나는 단정형 문체를 사용하세요.
3. 출력은 반드시 유효한 JSON 형식만 사용하세요.

출력 예시 (JSON):
[
  {{
   "contentId" : 문제 번호,
   "question": "질문",
   "correctAnswer": "정답",
   "answerExplanation": "해설"
  }}
]

뉴스 요약:
{summary}
"""
  res = llm_i.invoke(prompt_i)
  text = res.content.strip().replace("```json", "").replace("```", "")
  return json.loads(text)

def generate_advanced_e(i_quiz, summary):
    """I단계 10문항 중 5개를 자동 선정 후 E단계로 변환"""
    
    # === 1️. E단계 후보 선정 ===
    sel_prompt = f"""
    다음 I단계 문제 6개 중에서 E단계(고급 문어체)로 변환하기 적합한 5개를 선택하세요.
    - 정책·사회적 의미 중심
    - 단순 수치·날짜형 제외
    - JSON 배열로 질문만 출력 (예: ["질문1", "질문2", ...])

    문제 목록:
    {json.dumps([q["question"] for q in i_quiz], ensure_ascii=False, indent=2)}
    """
    sel_res = llm_i.invoke(sel_prompt)
    sel_text = sel_res.content.strip().replace("```json", "").replace("```", "")
    try:
        selected_questions = json.loads(sel_text)
    except json.JSONDecodeError:
        print("E단계 후보 선택 실패 — 랜덤으로 대체")
        selected_questions = random.sample([q["question"] for q in i_quiz], 5)

    # === 2️. E단계 변환 대상 필터 ===
    target_items = [q for q in i_quiz if q["question"] in selected_questions]

    # === 3️. E단계 문제 작성 ===
    prompt_e = f"""
당신은 뉴스 기반 학습 설계자이자 교육 평가 전문가입니다.
아래 추출된 I단계 문제 3개를 **고급 문어체 단답식(E 단계)** 문제 5개로 재작성하세요.

[어휘 및 문체 기준]
1. 문체는 공식적·논리적이며, 사실에 근거하되 사회적·정책적 의미를 암시해야 합니다.
2. **독해 속도를 늦추되, 사실적 근거는 기사에 명시된 내용만 사용**

[표현]
독해 속도를 늦추는 질문 예시 : 

- "정부가 어떤 제도를 발표했는가?"  
  → "정부가 최근 공표한 제도적 조치의 명칭은 무엇인가?"

- "이 사건이 발생한 원인은?"  
  → "해당 사안이 초래된 배경 요인은 무엇인가?"

[문항 구성 기준]
1. I단계의 정답은 그대로 유지
2. 정답은 반드시 **명사 한 단어**로 작성
3. 질문은 그 단어의 의미·맥락을 탐구하도록 하세요.

[해설 지침]
1. 50자 내외로 아래 두 요소를 모두 포함해야 함:
  - 정답의 사실적 근거 (기사 내용 기반)
  - 정답이 갖는 사회적·정책적·논리적 중요성
2. 모든 해설은 한 문장으로 작성하고, ‘~다.’로 끝나는 단정형 문체를 사용.

[세부 규칙]
1. question: 30-40자, 의문형 문어체
2. explanation: 최소 40자, 50자 내외
3. JSON 배열로 출력 
4. 기사 외 정보, 추측, 사견 절대 금지
5. 출력은 반드시 유효한 JSON 형식만 사용하세요.

출력 예시 (JSON):
[
  {{
    "contentId" : 문제 번호,
    "question": "질문",
    "correctAnswer": "정답",
    "answerExplanation": "해설"
  }}
]


[E 문제 후보]
{json.dumps(target_items, ensure_ascii=False, indent=2)}

[뉴스 요약]
{summary}

이제 위의 기준을 충실히 반영하여,
**단답형이지만 어휘·논리·맥락이 모두 한 단계 높은 고급 문어체 E단계 문제 5개**를 생성하세요.
"""
    e_res = llm_e.invoke(prompt_e)
    text = e_res.content.strip().replace("```json", "").replace("```", "")

    try:
        e_items = json.loads(text)
    except json.JSONDecodeError:
        print("JSON 파싱 실패 — LLM 응답 확인 필요")
        e_items = []

    # === 4️. 남은 I단계 5문항 ===
    remaining_i = [q for q in i_quiz if q["question"] not in selected_questions]

    # 각 단계별 contentId 재정렬 (1~5)
    for idx, q in enumerate(remaining_i, start=1):
        q["contentId"] = idx
    for idx, q in enumerate(e_items, start=1):
        q["contentId"] = idx

    print(f"변환된 E단계 문제 수: {len(e_items)}")
    print(f"남은 I단계 문제 수: {len(remaining_i)}")

    # 반드시 반환
    return e_items, remaining_i

# === 실행 ===
print("=== 뉴스 요약문 ===")
print(summary)

print("\n=== I단계 문제 생성 ===")
i_quiz = generate_quiz_i(summary)
print(json.dumps(i_quiz, ensure_ascii=False, indent=2))

print("\n=== E단계 문제 변환 ===")
e_quiz, remaining_i = generate_advanced_e(i_quiz, summary)

print("\n[E단계 결과]")
print(json.dumps(e_quiz, ensure_ascii=False, indent=2))

print("\n[남은 I단계 5문항]")
print(json.dumps(remaining_i, ensure_ascii=False, indent=2))

# === 저장 ===
SAVE_DIR = BASE_DIR / "data" / "quiz"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
today = datetime.now().strftime("%Y-%m-%d")

final_result = [
    {
        "contentType": "SHORT",
        "level": "I",
        "sourceUrl": sourceUrl,
        "contents": remaining_i
    },
    {
        "contentType": "SHORT",
        "level": "E",
        "sourceUrl": sourceUrl,
        "contents": e_quiz
    }
]

file_path = SAVE_DIR / f"{topic}_{course_id}_{session_id}_SHORT_IE_{today}.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(final_result, f, ensure_ascii=False, indent=2)

print(f"\n 전체 저장 완료 → {file_path.resolve()}")
print("(I단계 5문항, E단계 5문항 — 총 10문항)")