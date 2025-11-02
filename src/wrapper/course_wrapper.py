# === src/wrapper/course_wrapper.py ===
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
COURSE_FILTERED_DIR = BASE_DIR / "data" / "course_db" / "filtered"
QUIZ_AGGREGATED_DIR = BASE_DIR / "data" / "quiz" / "aggregated"

def build_course_package(topic: str):
    """
    1️. filtered 폴더의 코스 + 세션 정보 불러오기
    2️. aggregated 폴더의 퀴즈 통합 데이터 병합 (courseId + sessionId 기준)
    3️. 결과를 filtered/{topic}_full.json 으로 저장
    """
    course_path = COURSE_FILTERED_DIR / f"{topic}.json"
    quiz_path = QUIZ_AGGREGATED_DIR / f"{topic}_quizzes.json"
    output_path = COURSE_FILTERED_DIR / f"{topic}_full.json"

    # --- 1. 코스 로드 ---
    with open(course_path, "r", encoding="utf-8") as f:
        courses = json.load(f)

    # --- 2. 퀴즈 로드 ---
    with open(quiz_path, "r", encoding="utf-8") as f:
        quiz_data = json.load(f)

    # courseId + sessionId 기반 매핑
    quizzes_by_course_session = {
        f"{q['courseId']}_{q['sessionId']}": q["quizzes"]
        for q in quiz_data
    }

    # === STEP 순서 정의 ===
    STEP_ORDER_MAP = {
        "N": [
            ("SUMMARY_READING", 1),
            ("TERM_LEARNING", 2),
            ("CURRENT_AFFAIRS", 3),
            ("OX_QUIZ", 4),
            ("MULTIPLE_CHOICE", 5),
        ],
        "I": [
            ("SUMMARY_READING", 1),
            ("MULTIPLE_CHOICE", 2),
            ("SHORT_ANSWER", 3),
            ("SESSION_REFLECTION", 4),
        ],
        "E": [
            ("SUMMARY_READING", 1),
            ("SHORT_ANSWER", 2),
            ("SENTENCE_COMPLETION", 3),
            ("SESSION_REFLECTION", 4),
        ],
    }

    # --- 3. 코스 + 세션 + 퀴즈 병합 ---
    for course in courses:
        cid = course.get("courseId")
        for session in course.get("sessions", []):
            sid = session.get("sessionId")
            key = f"{cid}_{sid}"
            quiz_bundle = quizzes_by_course_session.get(key, {})
            if not quiz_bundle:
                continue

            level_map = {}
            for qtype, levels in quiz_bundle.items():
                for level, qcontent in levels.items():
                    if level not in level_map:
                        level_map[level] = []
                    order = next(
                        (o for q, o in STEP_ORDER_MAP.get(level, []) if q == qtype),
                        None,
                    )
                    if order:
                        level_map[level].append({
                            "stepOrder": order,
                            "contentType": qtype,
                            "contents": qcontent.get("contents", qcontent)
                        })

            session["quizzes"] = [
                {"level": level, "steps": sorted(steps, key=lambda x: x["stepOrder"])}
                for level, steps in level_map.items()
            ]

    # --- 4. filtered 안에 결과 저장 ---
    final_package = {"courses": courses}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_package, f, ensure_ascii=False, indent=2)

    print(f"[{topic}] 통합 패키지 생성 완료 → {output_path.resolve()}")
    return output_path