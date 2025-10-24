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

print(f"\n선택된 태그: {course_id}")
print(f"sessionId: {session_id}")
print(f"제목: {headline}\n")

# === 모델 설정 ===
llm_i = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)  # I단계
llm_e = ChatOpenAI(model="gpt-4o")

# === I단계 문제 생성 ===
def generate_quiz_i(summary: str):
    prompt_i = f"""
당신은 뉴스 기반 학습 퀴즈 생성 AI입니다.
아래 뉴스 내용을 참고하여 단답식 문제를 10개 만들어주세요.

[어휘 및 문체 기준]
1. 원문의 의미를 유지하되, 표현은 더 정제되고 분석적으로 바꿉니다.
2. 문어체를 사용하되, 지나치게 학술적이지 않게 자연스럽게 유지합니다.

[문항 구성 기준]
1. 뉴스의 핵심 사실(숫자, 인물, 정책명 등)을 직접 확인할 수 있는 문제
2. 육하원칙(언제·어디서·누가·무엇을)에 충실한 사실 기반 질문으로 구성
3. ‘왜’, ‘어떻게’ 등 원인·방법형 질문은 제외
4. 각 문항은 서로 다른 사실 요소를 다룸
5. 정답은 뉴스에 명시된 내용만 사용

[추가 지침]
1. 날짜, 연도, 수치, 통계, 인용문, 이름 등을 묻는 문제는 생성하지 마세요.

[세부 규칙]
1. question은 40자 이내, 한 문장
2. 정답은 한 단어 (뉴스에 명시된 내용만)
3. 해설은 50자 이내, 사실적 근거 중심
4. 출력은 반드시 유효한 JSON 형식만 사용하세요.

출력 예시 (JSON):
[
  {{
    "question": "질문 내용",
    "answers": [
      {{
        "text": "정답",
        "isCorrect": "true",
        "explanation": "해설"
      }}
    ]
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
    다음 I단계 문제 10개 중에서 E단계(고급 문어체)로 변환하기 적합한 5개를 선택하세요.
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
아래 추출된 I단계 문제 5개를 **고급 문어체 단답식(E 단계)** 문제 5개로 재작성하세요.

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
1. I단계의 정답(answer)은 그대로 유지
2. 정답은 한 단어
3. 질문은 그 단어의 의미·맥락을 탐구하도록 하세요.

[해설(explanation) 지침]
1. 해설은 단순히 ‘왜 정답인가’를 설명하는 수준을 넘어서야 함
2. 50자 내외로 아래 두 요소를 모두 포함해야 함:
  - 정답의 사실적 근거 (기사 내용 기반)
  - 정답이 갖는 사회적·정책적·논리적 중요성

[세부 규칙]
1. question: 문어체, 35~45자
2. explanation: 최소 45자, 50자 내외
3. JSON 배열로 출력 (camelCase 필드명 사용)
4. 기사 외 정보, 추측, 사견 절대 금지
5. 출력은 반드시 유효한 JSON 형식만 사용하세요.

출력 예시 (JSON):
[
  {{
    "question": "국방부가 방첩사 개편 일정을 공식화한 시점은 언제인가?",
    "answers": [
      {{
        "text": "2026년",
        "isCorrect": "true",
        "explanation": "국방부는 방첩사 개편을 내년까지 완료하겠다고 명시했다. 이는 조직 재편의 실행력과 투명성을 학습자가 인식하도록 유도하기 위함이다."
      }}
    ]
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
        "topic" : topic,
        "courseId": course_id,
        "sessionId": session_id,
        "contentType": "short",
        "level": "i",
        "items": remaining_i
    },
    {
        "topic" : topic,
        "courseId": course_id,
        "sessionId": session_id,
        "contentType": "short",
        "level": "e",
        "items": e_quiz
    }
]

file_path = SAVE_DIR / f"{topic}_{course_id}_short_ie_{today}.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(final_result, f, ensure_ascii=False, indent=2)

print(f"\n 전체 저장 완료 → {file_path.resolve()}")
print("(I단계 5문항, E단계 5문항 — 총 10문항)")