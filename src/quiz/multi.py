import os, json, random, numpy as np
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from langchain_openai import ChatOpenAI
from langchain.prompts import load_prompt
from langchain.prompts import ChatPromptTemplate
import yaml
from langchain.schema.runnable import RunnableMap, RunnableLambda
from select_session import select_session

# === 1️. 환경 변수 로드 ===
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === 2️. 세션 선택 ===
selected_session = select_session()
topic = selected_session["topic"]
course_id = selected_session["courseId"]            
session_id = selected_session.get("sessionId")
headline = selected_session.get("headline", "")
summary = selected_session.get("summary", "")
sourceUrl = selected_session.get("sourceUrl", "")

print(f"\n=== 세션 정보 ===")
print(f"코스 ID: {course_id}")
print(f"Session ID: {session_id}")
print(f"제목: {headline}")
print(f"요약문: {summary[:200]}...\n")

# === 3️. 모델 & 임베더 설정 ===
llm_n = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
llm_i = ChatOpenAI(model="gpt-4o", temperature=0.3)
llm_harder = ChatOpenAI(model="gpt-5")
embedder = SentenceTransformer("jhgan/ko-sroberta-multitask")

# === 4️. UTF-8 안전 YAML 로더 ===
def load_utf8_prompt(path: str):
    """UTF-8 안전하게 LangChain prompt YAML 로드"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # LangChain YAML은 "_type"과 "template"이 반드시 존재해야 함
    if "_type" not in data or "template" not in data:
        raise ValueError(f"잘못된 YAML 구조: {path}")

    return ChatPromptTemplate.from_template(data["template"])

prompt_fact_n = load_utf8_prompt("src/quiz/prompt/fact_n.yaml")
prompt_inference_i = load_utf8_prompt("src/quiz/prompt/inference_i.yaml")
prompt_harder_i = load_utf8_prompt("src/quiz/prompt/harder_i.yaml")

# === 5️. JSON 파서 ===
def parse_json_output(res):
    text = res.content.strip().replace("```json", "").replace("```", "")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print("JSON 파싱 오류 — 원문 출력:\n", text)
        return []

parse_node = RunnableLambda(parse_json_output)

# === 6️. 검증 (추론형 문제 전용) ===
def validate_inference_simple(candidates, summary, embedder, q_threshold=0.1, a_threshold=0.3):
    validated = []
    for cand in candidates:
        q = cand.get("question", "")
        options = cand.get("options", [])
        correct_label = cand.get("correctAnswer")
        correct_option = next((opt["text"] for opt in options if opt["label"] == correct_label), None)
        if not q or not correct_option:
            cand["validation"] = "데이터 누락"
            continue

        q_sim = util.cos_sim(embedder.encode(q), embedder.encode(summary)).item()
        a_sim = util.cos_sim(embedder.encode(correct_option), embedder.encode(summary)).item()
        score = round(q_sim * 0.4 + a_sim * 0.6, 2)

        cand.update({
            "question_sim": q_sim,
            "answer_sim": a_sim,
            "score": score,
            "validation": "통과" if (q_sim >= q_threshold and a_sim >= a_threshold) else "근거 부족"
        })

        if cand["validation"] == "통과":
            validated.append(cand)
    return validated

# === 7️. RunnableMap 파이프라인 구성 ===
quiz_pipeline = RunnableMap({
    "fact_n": prompt_fact_n | llm_n | parse_node,
    "inference_i": prompt_inference_i | llm_i | parse_node,
    "harder_i": (
        RunnableLambda(lambda x: {
            "n_quiz": json.dumps(x["fact_n"], ensure_ascii=False, indent=2),
            "summary": x["summary"]
        })
        | prompt_harder_i
        | llm_harder
        | parse_node
    ),
})

# === 8. 문제 생성  ===
def generate_all_quizzes(summary: str):
    print("=== 퀴즈 생성 시작 ===")

    # ① N단계 (사실형)
    fact_n = (prompt_fact_n | llm_n | parse_node).invoke({"summary": summary})
    print(f"N단계 완료: {len(fact_n)}문항 생성")

    # ② I단계 (추론형)
    inference_i = (prompt_inference_i | llm_i | parse_node).invoke({"summary": summary})
    print(f"I단계(추론형) 완료: {len(inference_i)}문항 생성")

    # ③ 검증
    print("\n=== 추론형 검증 ===")
    validated_i = validate_inference_simple(inference_i, summary, embedder)
    print(f"검증 통과 수: {len(validated_i)} / {len(inference_i)}")

    # ④ 부족분 계산
    num_needed = 5 - len(validated_i)
    harder_i = []
    if num_needed > 0:
        print(f"추론형 부족 → {num_needed}개 심화형(N단계 기반) 생성 중...")

        # 프롬프트 미리 채워서 문자열로 완성
        harder_prompt = prompt_harder_i.format_prompt(
            summary=summary,
            n_quiz=json.dumps(fact_n, ensure_ascii=False, indent=2),
            required_count=num_needed
        )

        # LLM 직접 호출 + JSON 파싱
        harder_response = llm_harder.invoke(harder_prompt.to_messages())
        harder_i = parse_json_output(harder_response)
        print(f"심화형 생성 완료: {len(harder_i)}문항")

    # ⑤ 최종 I단계 = 추론형(통과) + 심화형 보충
    total_i = validated_i + harder_i
    total_i = total_i[:5]

    for idx, q in enumerate(total_i, start=1):
        q["contentId"] = str(idx)
    print(f"최종 I단계 문제 수: {len(total_i)}")

    return {
        "fact_n": fact_n,       # 사실형 5문항
        "inference_i": total_i  # 추론형+심화형 합쳐서 5문항
    }

# === 8. 출력 포맷  ===
def format_quiz_output(data):
    """
    N단계, I단계, 심화형 모두 동일 포맷 유지.
    (I단계에는 추론형+심화형이 이미 통합되어 있음)
    """
    formatted = []

    for level, key in [("N", "fact_n"), ("I", "inference_i")]:
        items = data.get(key, [])
        cleaned = []
        for item in items:
            cleaned.append({
                "contentId": item.get("contentId"),
                "question": item.get("question"),
                "options": item.get("options"),
                "correctAnswer": item.get("correctAnswer"),
                "answerExplanation": item.get("answerExplanation")
            })
        formatted.append({
            "contentType": "MULTIPLE_CHOICE",
            "level": level,
            "sourceUrl": sourceUrl,
            "contents": cleaned
        })
    return formatted

# === 10. 저장 ===
def save_quiz_json(data, topic, course_id, session_id):
    BASE_DIR = Path(__file__).resolve().parents[2]
    SAVE_DIR = BASE_DIR / "data" / "quiz"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")

    file_path = SAVE_DIR / f"{topic}_{course_id}_{session_id}_MULTIPLE_NI_{today}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n 저장 완료: {file_path.resolve()}")

# === 11. 실행 ===
if __name__ == "__main__":
    all_quizzes = generate_all_quizzes(summary)
    formatted_output = format_quiz_output(all_quizzes)
    save_quiz_json(formatted_output, topic, course_id, session_id)
    print("=== 전체 완료 ===")