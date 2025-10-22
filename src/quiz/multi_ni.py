import os, json, random, numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from langchain_openai import ChatOpenAI
from select_session import select_session

# === 환경 변수 로드 ===
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH, override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === 세션 선택 ===
selected_session = select_session()
tags = selected_session["tags"]            
session_id = selected_session.get("sessionId")
headline = selected_session.get("headline", "")
summary = selected_session.get("summary", "")

print(f"\n선택된 태그: {tags}")
print(f"sessionId: {session_id}")
print(f"제목: {headline}\n")

# === 모델 설정 ===
llm_n = ChatOpenAI(model="gpt-4o", temperature=0.3)  # N단계
llm_i = ChatOpenAI(model="gpt-5")
embedder = SentenceTransformer("jhgan/ko-sroberta-multitask")

# === N단계 문제 생성 ===
def generate_quiz_n(summary: str):
    prompt_n = f"""
당신은 뉴스 기반 학습 퀴즈 생성 AI입니다.
아래 뉴스 내용을 참고하여 **기초(N단계)** 다지선다 문제 5개를 만들어주세요.

🎯 목표:
- 뉴스의 핵심 사실을 확인할 수 있는 문제 5개
- 각 문제는 서로 다른 사실을 다뤄야 함

📘 규칙
- level="n"
- 질문 45자 내외
- 선다 15자 내외, 명사/구 단위
- 선다 중 1개만 정답
- 뉴스 요약의 사실만 사용
- 해설은 50자 이내, 모두 다르게 작성
- 각 보기별 해설은 간결하고 모두 달라야 함

출력은 반드시 유효한 JSON 형식만 사용하세요.
📘 출력 형식(JSON 배열):
[
  {{
    "question": "질문 내용",
    "answers": [
      {{"text": "선다1", "isCorrect": false, "explanation": "해설1"}},
      {{"text": "선다2", "isCorrect": true, "explanation": "해설2"}},
      {{"text": "선다3", "isCorrect": false, "explanation": "해설3"}},
      {{"text": "선다4", "isCorrect": false, "explanation": "해설4"}}
    ]
  }}
]

뉴스 요약:
{summary}
"""
    res = llm_n.invoke(prompt_n)
    text = res.content.strip().replace("```json", "").replace("```", "")
    return json.loads(text)

def generate_quiz_i(n_quiz, summary):
    prompt_i = f"""
당신은 뉴스 학습용 퀴즈를 설계하는 전문가입니다.
다음은 **N단계(기초)** 문제입니다.
이를 바탕으로 **I단계(심화)** 문제를 만들어주세요.

🎯 목표
- 원문의 의미는 유지하되, 표현을 한층 더 정제하고 분석적으로 바꿉니다.
- 단, E단계(확장·비판 단계)로 이어질 수 있도록, **이해력·추론력 중심의 중간 난이도**로 설계하세요.
- 문장은 문어체를 사용하되, **지나치게 학술적이거나 추상적 표현은 피합니다.**
- **질문은 기사 속 인과관계·핵심 논점·의미 변화를 중심으로 구성합니다.**
- **선지의 단어 표현은 한 단계 고급화**하되, 15자 내외로 간결하게 유지.
- **오답은 정답과 유사한 개념·시점·어휘**로 구성하되, 의미가 미묘하게 달라야 합니다.
- **정답은 기사 근거를 기반으로, 오답은 자주 혼동되는 맥락**을 반영하세요.

📘 세부 규칙
- level: "i"
- basedOn: 원본 질문
- question: 최소 40자, 45자 내외, 문어체
  (예: “~로 분석된다”, “~을 근거로 해석할 수 있다”)
- answers: 4개 (1개 정답, 3개 오답)
- 선지는 15자 내외, 명사/구 단위
- 해설(explanation): 최소 40자, 50자 내외, 명료한 한 문장, 정답의 근거와 오답의 차이를 논리적으로 설명
- 출력은 반드시 **유효한 JSON 배열** 형식으로만

입력 데이터:
N단계 문제:
{json.dumps(n_quiz, ensure_ascii=False, indent=2)}

뉴스 요약:
{summary}

출력은 반드시 아래 형식으로만 작성하세요.
[
  {{
    "basedOn": "원본 질문",
    "level": "i",
    "question": "심화 문어체 질문",
    "answers": [
      {{"text": "선지1", "isCorrect": false, "explanation": "해설1"}},
      {{"text": "선지2", "isCorrect": true, "explanation": "해설2"}},
      {{"text": "선지3", "isCorrect": false, "explanation": "해설3"}},
      {{"text": "선지4", "isCorrect": false, "explanation": "해설4"}}
    ]
  }}
]
"""

    res = llm_i.invoke(prompt_i)
    text = res.content.strip().replace("```json", "").replace("```", "")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print("JSON 파싱 오류 — 원문 출력:\n", text)
        return []

# === 검증 및 점수화 ===
def validate_and_score(candidates, summary):
    validated = []
    for cand in candidates:
        q = cand["question"]
        correct = next((opt["text"] for opt in cand["answers"] if opt["isCorrect"]), None)
        wrongs = [opt["text"] for opt in cand["answers"] if not opt["isCorrect"]]

        if util.cos_sim(embedder.encode(q), embedder.encode(summary)).item() < 0.3:
            cand["validation"] = "기사 근거 약함"
            continue

        sims = [util.cos_sim(embedder.encode(correct), embedder.encode(w)).item() for w in wrongs]
        mean_sim = np.mean(sims)
        if mean_sim > 0.8:
            cand["validation"] = f"오답 유사도 과다 ({mean_sim:.2f})"
            continue

        clarity = 1 if len(q) > 40 else 0.7
        grounding = 1 if any(word in summary for word in correct.split()) else 0.7
        diversity = 1 - mean_sim
        score = round((clarity * 0.3 + grounding * 0.3 + diversity * 0.4), 2)
        cand["score"] = score
        cand["validation"] = "통과" if score >= 0.75 else "점수 낮음"
        if score >= 0.75:
            validated.append(cand)
    return validated

# === 보기 섞기 ===
def shuffle_quiz_answers(quiz_list):
    for quiz in quiz_list:
        if "answers" in quiz:
            random.shuffle(quiz["answers"])
    return quiz_list

# === 실행 ===
print("=== 뉴스 요약문 ===")
summary = selected_session["summary"]
print(summary)

print("=== N단계 문제 생성 ===")
n_quiz = generate_quiz_n(summary)
print(json.dumps(n_quiz, ensure_ascii=False, indent=2))

print("\n=== I단계 문제 생성 ===")
i_quiz = generate_quiz_i(n_quiz, summary)
print(json.dumps(i_quiz, ensure_ascii=False, indent=2))

print("\n=== 검증 및 점수 계산 ===")
validated_quiz = validate_and_score(i_quiz, summary)
print(json.dumps(validated_quiz, ensure_ascii=False, indent=2))

print("\n=== 보기 순서 랜덤화 ===")
shuffled_quiz = shuffle_quiz_answers(validated_quiz)
print(json.dumps(shuffled_quiz, ensure_ascii=False, indent=2))

# === 저장 ===
SAVE_DIR = BASE_DIR / "data" / "quiz"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
today = datetime.now().strftime("%Y-%m-%d")

def strip_debug_info(quiz_list):
    clean_list = []
    for q in quiz_list:
        # validation, score, basedOn 등 디버그용 필드 제거
        q_clean = {
            "question": q.get("question"),
            "answers": q.get("answers"),
        }
        # 해설 필드 유지 (있을 경우)
        if "explanation" in q:
            q_clean["explanation"] = q["explanation"]
        clean_list.append(q_clean)
    return clean_list

clean_i_quiz = strip_debug_info(i_quiz)
clean_n_quiz = strip_debug_info(n_quiz)

# === 저장 ===
final_result = [
    {
        "sessionId": session_id,
        "tags": tags,
        "contentType": "multi",
        "level": "n",
        "items": clean_n_quiz,
    },
    {
        "sessionId": session_id,
        "tags": tags,
        "contentType": "multi",
        "level": "i",
        "items": clean_i_quiz,
    },
]

file_path = SAVE_DIR / f"{tags}_multi_ni_{today}.json"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(final_result, f, ensure_ascii=False, indent=2)

print(f"\nN단계 + I단계 통합 저장 완료 → {file_path.resolve()}")
print("=== 전체 완료 ===")