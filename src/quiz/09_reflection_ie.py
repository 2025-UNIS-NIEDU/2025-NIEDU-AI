import os, json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# === 환경 변수 로드 ===
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH, override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === 경로 설정 ===
QUIZ_DIR = BASE_DIR / "data" / "quiz"

BASE_DIR = Path(__file__).resolve().parents[2]
QUIZ_DIR = BASE_DIR / "data" / "quiz"
topic = "politics"
today = datetime.now().strftime("%Y-%m-%d")

# === 한 번에 모든 관련 파일 로드 ===
patterns = [
    f"{topic}_multi_ni_*.json",
    f"{topic}_short_ie_*.json",
    f"{topic}_completion_e_*.json"
]

all_blocks = []

for pattern in patterns:
    for file in QUIZ_DIR.glob(pattern):
        print(f"{file.name}")
        with open(file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                all_blocks.extend(data)
            except json.JSONDecodeError:
                print(f"JSON 오류: {file}")
                continue

# === 메타데이터 기반으로 분리 ===
i_items = []
e_items = []

course_id = None
session_id = None

for block in all_blocks:
    if not isinstance(block, dict):
        continue
    if course_id is None:
        course_id = block.get("courseId")
    if session_id is None:
        session_id = block.get("sessionId")

for block in all_blocks:
    if not isinstance(block, dict):
        print("문자열 block 감지:", block)
        continue
    level = block.get("level")
    if level == "i":
        i_items.extend(block.get("items", []))
    elif level == "e":
        e_items.extend(block.get("items", []))

print(f"I단계 {len(i_items)}개, E단계 {len(e_items)}개 로드 완료.")

# === 예시: 회고형 프롬프트 입력용 ===
i_questions = [q["question"] for q in i_items]
e_questions = [q["question"] for q in e_items]

print("\n=== I단계 질문 미리보기 ===")
for q in i_questions[:3]:
    print("•", q)

print("\n=== E단계 질문 미리보기 ===")
for q in e_questions[:3]:
    print("•", q)


# === 모델 설정 ===
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# === 회고 단계 문제 생성 ===
def generate_quiz_i(i_quiz, summary=""):
    prompt_i = f"""
당신은 뉴스 기반 학습 퀴즈 생성 AI입니다.
아래는 동일 뉴스에 대해 이미 생성된 **I단계 문제** 목록입니다.
이 문제들을 분석하고, 이를 바탕으로 학습자가 사건을 되돌아볼 수 있는 **회고형 질문 1개**를 만드세요.

🎯 목표:
- 주어진 I단계 문제들의 맥락을 반영하여, 뉴스의 본질적 의미를 성찰하도록 유도
- 정답이 정해져 있지 않은 사고형 질문
- 단문 문어체 (20자 내외)
- 반드시 JSON 형식을 따를 것

⚙️ 출력 예시(JSON 배열):
[
  {{
    "question": "이번 사안이 사회에 남긴 교훈은?"
  }}
]

=== I단계 문제 목록 ===
{json.dumps(i_quiz, ensure_ascii=False, indent=2)}

=== 뉴스 요약 ===
{summary}
"""
    res = llm.invoke(prompt_i)
    text = res.content.strip().replace("```json", "").replace("```", "")
    return json.loads(text)

def generate_quiz_e(e_quiz, summary=""):
    prompt_e = f"""
당신은 뉴스 기반 학습 설계 전문가입니다.
아래는 동일 뉴스에 대해 이미 생성된 **E단계 문제** 목록입니다.
이를 분석하여, 학습자가 사건의 원인·결과·의미를 성찰할 수 있는 **회고형 질문 1개**를 새로 작성하세요.

🎯 목표:
- 뉴스의 사회적 함의, 제도적 문제, 혹은 시사점을 성찰하도록 유도
- 정답이 없고, 사고를 요구하는 질문
- 문어체 20자 내외
- 반드시 JSON 형식을 따를 것

⚙️ 출력 예시(JSON 배열):
[
  {{
    "question": "이 개편 논의가 제도 신뢰에 미친 영향은?"
  }}
]

=== E단계 문제 목록 ===
{json.dumps(e_quiz, ensure_ascii=False, indent=2)}

=== 뉴스 요약 ===
{summary}
"""
    res = llm.invoke(prompt_e)
    text = res.content.strip().replace("```json", "").replace("```", "")
    return json.loads(text)

# === 실행 ===
print("\n=== I단계 회고 문제 생성 ===")
i_reflection = generate_quiz_i(i_items)
print(json.dumps(i_reflection, ensure_ascii=False, indent=2))

print("\n=== E단계 회고 문제 생성 ===")
e_reflection = generate_quiz_e(e_items)
print(json.dumps(e_reflection, ensure_ascii=False, indent=2))

# === 저장 ===
SAVE_DIR = QUIZ_DIR
SAVE_DIR.mkdir(parents=True, exist_ok=True)

final_result = [
    {
        "courseId": course_id,
        "sessionId" : session_id,
        "topic": topic,
        "contentType": "reflection",
        "level": "i",
        "items": i_reflection
    },
    {
        "courseId": course_id,
        "sessionId" : session_id,
        "topic": topic,
        "contentType": "reflection",
        "level": "e",
        "items": e_reflection
    }
]

file_path = SAVE_DIR / f"{topic}_reflection_ie_{today}.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(final_result, f, ensure_ascii=False, indent=2)

print(f"\n I/E 회고형 문제 저장 완료 → {file_path.resolve()}")