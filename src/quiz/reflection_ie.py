import os, json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from select_session import select_session

# 환경 변수 및 경로 설정
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH, override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

QUIZ_DIR = BASE_DIR / "data" / "quiz"
today = datetime.now().strftime("%Y-%m-%d")

# 1️. 세션 선택
selected_session = select_session()
topic = selected_session["topic"]
courseId = selected_session["courseId"]
sessionId = selected_session["sessionId"]

# 2️. 선택된 세션 기반으로 파일 자동 필터링
prefix = f"{topic}_{courseId}_{sessionId}_"
target_files = sorted([f for f in QUIZ_DIR.glob(f"{prefix}*.json")])

if not target_files:
    print(f"{prefix} 관련 JSON 파일이 없습니다.")
    exit()

print("\n=== 자동 감지된 퀴즈 파일 목록 ===")
for idx, f in enumerate(target_files, 1):
    print(f"{idx:2d}. {f.name}")

# 3️. 모든 관련 파일 로드
all_blocks = []
for file_path in target_files:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_blocks.extend(data if isinstance(data, list) else [data])
        print(f"불러옴 → {file_path.name}")
    except json.JSONDecodeError:
        print(f"JSON 오류: {file_path.name}")

print(f"\n총 {len(all_blocks)}개 블록 로드 완료\n")

# 4️. I/E 단계 분리
i_items, e_items = [], []
for block in all_blocks:
    if not isinstance(block, dict):
        continue
    level = block.get("level", "")
    if level == "i":
        i_items.extend(block.get("items", []))
    elif level == "e":
        e_items.extend(block.get("items", []))

print(f"I단계 {len(i_items)}개, E단계 {len(e_items)}개 로드 완료\n")

# 5️. 질문 미리보기
print("=== I단계 질문 미리보기 ===")
for q in [q.get("question") for q in i_items[:3]]:
    print("•", q)

print("\n=== E단계 질문 미리보기 ===")
for q in [q.get("question") for q in e_items[:3]]:
    print("•", q)

# 6️. LLM 설정
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

# 7️. 회고형 문제 생성 함수
def generate_reflection(level, quiz_items):
    # 단계별 설명 정의
    if level == "i":
        phase_desc = (
            "이해 및 분석 중심의 회고형 질문을 만드세요. "
            "뉴스의 사실과 인과관계를 되짚으며, 사건의 본질이나 의도를 탐구하는 데 초점을 둡니다. "
            "학습자가 사건의 의미를 스스로 분석하도록 유도해야 합니다."
        )
    else:  # E 단계
        phase_desc = (
            "비판적·확장적 사고를 유도하는 회고형 질문을 만드세요. "
            "사건이 사회, 제도, 가치, 윤리 등에 미치는 영향을 고려하고, "
            "다른 시각에서 재해석하거나 대안을 고민하도록 이끕니다."
        )

    # 프롬프트 구성
    prompt = f"""
당신은 뉴스 기반 학습 퀴즈 생성 AI입니다.

뉴스의 내용을 회고적으로 이해하도록 돕는 **{level.upper()}단계 회고형 질문 1개**를 만드세요.
{phase_desc}

아래에는 이미 생성된 퀴즈 목록이 주어집니다.
이들을 참고해 “이 사건에 대해 어떤 점을 되돌아봐야 하는가”에 초점을 맞추세요.

규칙:
- 단문 문어체 (약 20자 내외)
- 질문에는 반드시 사건의 맥락이 암시되어야 함
- {("원인·의도·의미 중심" if level == "i" else "가치·함의·대안 중심")}
- 반드시 JSON 형식으로 출력 (배열 형태)
- 예시:
  [{{"question": "{'정부 개편안의 추진 배경은 무엇인가?' if level == 'i' else '정부 개편안이 사회적 신뢰에 미친 영향은?'}"}}]

=== {level.upper()}단계 문제 목록 ===
{json.dumps(quiz_items, ensure_ascii=False, indent=2)}
"""

    # LLM 호출 및 결과 처리
    res = llm.invoke(prompt)
    text = res.content.strip().replace("```json", "").replace("```", "")
    return json.loads(text)

# 9.️ 회고형 문제 생성
print("\n=== I단계 회고 문제 생성 ===")
i_reflection = generate_reflection("i", i_items)
print(json.dumps(i_reflection, ensure_ascii=False, indent=2))

print("\n=== E단계 회고 문제 생성 ===")
e_reflection = generate_reflection("e", e_items)
print(json.dumps(e_reflection, ensure_ascii=False, indent=2))

# 10. 결과 저장
final_result = [
    {
        "topic": topic,
        "courseId": courseId,
        "sessionId": sessionId,
        "contentType": "reflection",
        "level": "i",
        "items": i_reflection
    },
    {
        "topic": topic,
        "courseId": courseId,
        "sessionId": sessionId,
        "contentType": "reflection",
        "level": "e",
        "items": e_reflection
    }
]

file_path = QUIZ_DIR / f"{topic}_{courseId}_{sessionId}_reflection_ie_{today}.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(final_result, f, ensure_ascii=False, indent=2)

print(f"\n회고 문제 최종 저장 완료 → {file_path.resolve()}")