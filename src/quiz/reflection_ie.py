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
today = datetime.now().strftime("%Y-%m-%d")

target_names = [
    "['economy', '#금융']_multi_ni_2025-10-22.json",
    "['economy', '#금융']_short_ie_2025-10-22.json",
    "['economy', '#금융']_completion_e_2025-10-22.json",
    "['economy', '#금융']_reflection_ie_2025-10-22.json"
]

# === 폴더 내 실제 파일 목록 ===
all_files = [f.name for f in QUIZ_DIR.glob("*.json")]
print("현재 quiz 폴더 내 JSON 파일:")
for f in all_files:
    print("  -", f)

# === 정확히 파일명 비교해서 로드 ===
all_blocks = []
for name in target_names:
    match = [f for f in all_files if f == name]
    if not match:
        print(f"파일 없음: {name}")
        continue
    file_path = QUIZ_DIR / match[0]
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            all_blocks.extend(data if isinstance(data, list) else [data])
            print(f"불러옴 → {name}")
        except json.JSONDecodeError:
            print(f"JSON 오류: {name}")

print(f"\n총 {len(all_blocks)}개 블록 로드 완료")

# === I/E 단계 분리 ===
i_items, e_items = [], []
i_meta, e_meta = {}, {}  # 메타데이터 저장용
for block in all_blocks:
    if not isinstance(block, dict):
        continue
    level = block.get("level", "")
    if level == "i":
        i_items.extend(block.get("items", []))
        if not i_meta:
            i_meta = {
                "sessionId": block.get("sessionId"),
                "tags": block.get("tags")
            }
    elif level == "e":
        e_items.extend(block.get("items", []))
        if not e_meta:
            e_meta = {
                "sessionId": block.get("sessionId"),
                "tags": block.get("tags")
            }

print(f"\nI단계 {len(i_items)}개, E단계 {len(e_items)}개 로드 완료")

# === 질문 미리보기 ===
print("\n=== I단계 질문 미리보기 ===")
for q in [q.get("question") for q in i_items[:3]]:
    print("•", q)

print("\n=== E단계 질문 미리보기 ===")
for q in [q.get("question") for q in e_items[:3]]:
    print("•", q)

# === 모델 설정 ===
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# === 회고형 질문 생성 ===
def generate_reflection(level, quiz_items):
    prompt = f"""
당신은 뉴스 기반 학습 퀴즈 생성 AI입니다.

뉴스의 내용을 회고적으로 이해하도록 돕는 **{level.upper()}단계 회고형 질문 1개**를 만드세요.
아래에는 이미 생성된 퀴즈 목록이 주어집니다. 이들을 참고해 “이 사건에 대해 어떤 점을 되돌아봐야 하는가”에 초점을 맞추세요.

🎯 규칙:
- 단문 문어체 (약 20자 내외)
- 질문에는 반드시 사건의 맥락이 암시되어야 함
- 질문은 개인의 생각뿐 아니라 사회적 시각, 타인의 입장, 제도적 관점 등 다양한 해석이 가능해야 함
- 반드시 JSON 형식으로 출력 (배열 형태)
- 예시:
  [{{"question": "정부 개편안이 공정성 논의에 미친 영향은?"}}]

=== {level.upper()}단계 문제 목록 ===
{json.dumps(quiz_items, ensure_ascii=False, indent=2)}
"""
    res = llm.invoke(prompt)
    text = res.content.strip().replace("```json", "").replace("```", "")
    return json.loads(text)

# === 실행 ===
print("\n=== I단계 회고 문제 생성 ===")
i_reflection = generate_reflection("i", i_items)
print(json.dumps(i_reflection, ensure_ascii=False, indent=2))

print("\n=== E단계 회고 문제 생성 ===")
e_reflection = generate_reflection("e", e_items)
print(json.dumps(e_reflection, ensure_ascii=False, indent=2))

# === 저장 ===
final_result = [
    {
        "sessionId": i_meta.get("sessionId"),
        "tags": i_meta.get("tags"),
        "contentType": "reflection",
        "level": "i",
        "items": i_reflection
    },
    {
        "sessionId": e_meta.get("sessionId"),
        "tags": e_meta.get("tags"),
        "contentType": "reflection",
        "level": "e",
        "items": e_reflection
    }
]

save_path = QUIZ_DIR / "['economy','#금융']_reflection_ie_merged_2025-10-22.json"
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(final_result, f, ensure_ascii=False, indent=2)

print(f"\n💾 회고 문제 최종 저장 완료 → {save_path.resolve()}")