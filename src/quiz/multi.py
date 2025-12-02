import os, json, random, logging, numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate 
import yaml
from langchain.schema.runnable import RunnableLambda
from quiz.select_session import select_session

logger = logging.getLogger(__name__)

def generate_multi_choice_quiz(selected_session=None):
    """N·I단계 객관식 퀴즈 자동 생성"""

    # === 1️. 환경 변수 로드 ===
    load_dotenv(override=True)

    # === 2️. 세션 선택 ===
    if selected_session is None:
        selected_session = select_session()

    topic = selected_session["topic"]
    course_id = selected_session["courseId"]
    session_id = selected_session.get("sessionId")
    headline = selected_session.get("headline", "")
    summary = selected_session.get("summary", "")
    sourceUrl = selected_session.get("sourceUrl", "")

    logger.info(f"[{topic}] 코스 {course_id} 세션 {session_id} MULTIPLE_CHOICE 생성 시작")

    # === 3️. 모델 & 임베더 설정 ===
    llm_n = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_i = ChatOpenAI(model="gpt-4o", temperature=0.3)
    llm_harder = ChatOpenAI(model="gpt-5")
    embedder = SentenceTransformer("jhgan/ko-sroberta-multitask")

    # === 4️. UTF-8 안전 YAML 로더 ===
    def load_utf8_prompt(path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if "_type" not in data or data["_type"] not in {"prompt", "prompt_template"}:
            raise ValueError(f"{path}: '_type'이 'prompt' 또는 'prompt_template' 이어야 합니다.")
        if "template" not in data:
            raise ValueError(f"{path}: 'template' 필드가 없습니다.")

        return PromptTemplate.from_template(data["template"])

    prompt_fact_n = load_utf8_prompt("src/quiz/prompt/fact_n.yaml")
    prompt_inference_i = load_utf8_prompt("src/quiz/prompt/inference_i.yaml")
    prompt_harder_i = load_utf8_prompt("src/quiz/prompt/harder_i.yaml")

    # === 5️. JSON 파서 ===
    def parse_json_output(res):
        text = res.content.strip().replace("```json", "").replace("```", "")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"[{topic}] JSON 파싱 오류 발생")
            return []

    parse_node = RunnableLambda(parse_json_output)

    # === 6️. 검증 ===
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
            score = round(q_sim * 0.3 + a_sim * 0.7, 2)
            cand.update({
                "question_sim": q_sim,
                "answer_sim": a_sim,
                "score": score,
                "validation": "통과" if (q_sim >= q_threshold and a_sim >= a_threshold) else "근거 부족"
            })
            if cand["validation"] == "통과":
                validated.append(cand)
        return validated

    # === 7. 문제 생성 ===
    def generate_all_quizzes(summary: str):
        logger.info(f"[{topic}] N·I단계 퀴즈 생성 중...")

        fact_n = (prompt_fact_n | llm_n | parse_node).invoke({"summary": summary})
        inference_i = (prompt_inference_i | llm_i | parse_node).invoke({"summary": summary})
        validated_i = validate_inference_simple(inference_i, summary, embedder)

        num_needed = 5 - len(validated_i)
        harder_i = []
        if num_needed > 0:
            logger.info(f"[{topic}] 추론형 부족 → {num_needed}개 심화형 생성 중")
            harder_prompt = prompt_harder_i.format(
                summary=summary,
                n_quiz=json.dumps(fact_n, ensure_ascii=False, indent=2),
                required_count=num_needed
            )
            harder_response = llm_harder.invoke(harder_prompt)
            harder_i = parse_json_output(harder_response)

        total_i = (validated_i + harder_i)[:5]
        
        # === 8. 정답 라벨 균등 배열 생성 ===
        total_questions = len(fact_n) + len(total_i)

        balanced_labels = ["A","B","C","D"] * ((total_questions // 4) + 1)
        random.shuffle(balanced_labels)
        balanced_labels = balanced_labels[:total_questions]

        balanced_index = 0

        # === 정답 위치 균등화 셔플 ===
        def assign_balanced_label(q, forced_label):
            options = q["options"]
            correct_label = q["correctAnswer"]
            
            # 정답 텍스트 찾아서
            correct_text = next(
                opt["text"] for opt in options if opt["label"] == correct_label
            )

            # 오답 텍스트
            distractors = [opt["text"] for opt in options if opt["text"] != correct_text]
            random.shuffle(distractors)

            # 새 옵션 생성
            labels = ["A","B","C","D"]
            new_opts = []

            for label in labels:
                if label == forced_label:
                    new_opts.append({"label": label, "text": correct_text})
                else:
                    new_opts.append({"label": label, "text": distractors.pop()})

            q["options"] = new_opts
            q["correctAnswer"] = forced_label
            return q

        # N단계
        for q in fact_n:
            forced_label = balanced_labels[balanced_index]
            assign_balanced_label(q, forced_label)
            balanced_index += 1

        # I단계
        for q in total_i:
            forced_label = balanced_labels[balanced_index]
            assign_balanced_label(q, forced_label)
            balanced_index += 1

        # contentId 재정렬
        for idx, q in enumerate(total_i, start=1):
            q["contentId"] = idx

        return {"fact_n": fact_n, "inference_i": total_i}

    # === 9. 출력 포맷 — LLM 안전 append 방식으로 변경 ===
    def format_quiz_output(data):
        formatted = []

        for level, key in [("N", "fact_n"), ("I", "inference_i")]:
            items = data.get(key, [])
            contents = []

            for idx, item in enumerate(items, start=1):

                # 안전 보정
                question = item.get("question", "").strip()
                options = item.get("options", [])
                correct_label = item.get("correctAnswer", "").strip()
                explanation = item.get("answerExplanation", "")

                # --- 옵션 보정 ---
                if not isinstance(options, list):
                    options = []

                # 옵션 4개 미만이면 채우기
                while len(options) < 4:
                    options.append({"label": None, "text": "보기를 찾을 수 없음"})

                # --- 정답 보정 ---
                # 이제 correct_text를 “보정된 options”에서 찾는다.
                correct_text = None
                for opt in item.get("options", []):
                    if opt.get("label") == correct_label:
                        correct_text = opt.get("text", "").strip()

                new_correct_label = None
                if correct_text:
                    for opt in options:
                        if opt["text"].strip() == correct_text:
                            new_correct_label = opt["label"]
                            break

                # 정답이 없을 경우 → fallback 없음 → 그냥 None으로 둔다

                contents.append({
                    "contentId": idx,  
                    "question": question or "",
                    "options": options or "",
                    "correctAnswer": new_correct_label or "",
                    "answerExplanation": explanation or "",
                    "sourceUrl": sourceUrl
                })

            formatted.append({
                "contentType": "MULTIPLE_CHOICE",
                "level": level,
                "sourceUrl": sourceUrl,
                "contents": contents
            })

        return formatted

    # === 10. 저장 ===
    def save_quiz_json(data, topic, course_id, session_id):
        BASE_DIR = Path(__file__).resolve().parents[2]
        SAVE_DIR = BASE_DIR / "data" / "quiz"
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")

        for quiz in data:
            level = str(quiz.get("level", "")).upper().strip()
            if not level:
                continue
            file_name = f"{topic}_{course_id}_{session_id}_MULTIPLE_CHOICE_{level}_{today}.json"
            file_path = SAVE_DIR / file_name
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump([quiz], f, ensure_ascii=False, indent=2)
            logger.info(f"[{topic}] {level}단계 퀴즈 저장 완료 → {file_path.name}")

    try:
        all_quizzes = generate_all_quizzes(summary)
        formatted_output = format_quiz_output(all_quizzes)
        save_quiz_json(formatted_output, topic, course_id, session_id)
        logger.info(f"[{topic}] 세션 {session_id} 퀴즈 생성 완료")
    except Exception as e:
        logger.error(f"[{topic}] 퀴즈 생성 중 오류 발생: {e}", exc_info=True)

# === 실행 ===
if __name__ == "__main__":
    generate_multi_choice_quiz()