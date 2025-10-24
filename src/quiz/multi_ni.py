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

print(f"\n=== 세션 정보 ===")
print(f"코스 ID: {course_id}")
print(f"Session ID: {session_id}")
print(f"제목: {headline}")
print(f"요약문: {summary[:200]}...\n")

# === 3️. 모델 & 임베더 설정 ===
llm_n = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
llm_i = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
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
def validate_inference_simple(candidates, summary, embedder, q_threshold=0.3, a_threshold=0.6):
    validated = []
    for cand in candidates:
        q = cand.get("question", "")
        answers = cand.get("answers", [])
        correct = next((a["text"] for a in answers if a.get("isCorrect")), None)
        if not q or not correct:
            cand["validation"] = "데이터 누락"
            continue

        q_sim = util.cos_sim(embedder.encode(q), embedder.encode(summary)).item()
        a_sim = util.cos_sim(embedder.encode(correct), embedder.encode(summary)).item()
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

# === 8️. 전체 실행 함수 ===
def generate_all_quizzes(summary: str):
    print("=== 퀴즈 생성 시작 ===")

    # 1️. N단계 (사실형)
    fact_n = (prompt_fact_n | llm_n | parse_node).invoke({"summary": summary})
    print("N단계 완료")

    # 2️. I단계 (추론형)
    inference_i = (prompt_inference_i | llm_i | parse_node).invoke({"summary": summary})
    print("I단계(추론형) 완료")

    # 3️. 검증 (추론형 문제 유효성)
    print("\n=== 추론형 검증 ===")
    validated_i = validate_inference_simple(inference_i, summary, embedder)
    print(f"검증 통과 수: {len(validated_i)} / {len(inference_i)}")

    # 4️. I단계 (심화형) - 검증된 결과 + 기초 문제 활용
    harder_input = {
        "n_quiz": json.dumps(fact_n, ensure_ascii=False, indent=2),
        "i_quiz": json.dumps(validated_i, ensure_ascii=False, indent=2),
        "summary": summary,
    }
    harder_i = (prompt_harder_i | llm_harder | parse_node).invoke(harder_input)
    print("I단계(심화형) 완료")

    return {
        "fact_n": fact_n,
        "inference_i": validated_i,
        "harder_i": harder_i,
    }

# === 9. 출력 포맷 변환 ===
def format_quiz_output(data, topic, course_id, session_id):
    """
    LangChain 생성 결과(fact_n, inference_i, harder_i)를
    최종 표준 출력 포맷으로 변환.
    """
    formatted = []

    # === N단계 ===
    formatted.append({
        "topic": topic,
        "courseId": course_id,
        "sessionId": session_id,
        "contentType": "multi",
        "level": "n",
        "items": data.get("fact_n", [])
    })

    # === I단계 ===
    i_items = data.get("inference_i") or data.get("harder_i", [])
    cleaned_i = []

    for item in i_items:
        cleaned_i.append({
            "question": item.get("question"),
            "answers": item.get("answers"),
        })

    formatted.append({
        "topic": topic,
        "courseId": course_id,
        "sessionId": session_id,
        "contentType": "multi",
        "level": "i",
        "items": cleaned_i
    })

    return formatted

# === 10. 저장 ===
def save_quiz_json(data, topic, course_id, session_id):
    BASE_DIR = Path(__file__).resolve().parents[2]
    SAVE_DIR = BASE_DIR / "data" / "quiz"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")

    file_path = SAVE_DIR / f"{topic}_{course_id}_{session_id}_multi_ni_{today}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n 저장 완료: {file_path.resolve()}")

# === 11. 실행 ===
if __name__ == "__main__":
    all_quizzes = generate_all_quizzes(summary)
    formatted_output = format_quiz_output(all_quizzes, topic, course_id, session_id)
    save_quiz_json(formatted_output, topic, course_id, session_id)
    print("=== 전체 완료 ===")