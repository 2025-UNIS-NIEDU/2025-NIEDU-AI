import json
import csv
from pathlib import Path

PACKAGE_DIR = Path("data/quiz/package")
OUTPUT_DIR = Path("data/quiz/csv")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STEP_ORDER_MAP = {
    "N": [
        ("SUMMARY_READING", 1),
        ("TERM_LEARNING", 2),
        ("CURRENT_AFFAIRS", 3),
        ("OX_QUIZ", 4),
        ("MULTIPLE_CHOICE", 5),
    ],
    "I": [
        ("ARTICLE_READING", 1),
        ("SUMMARY_READING", 2),
        ("MULTIPLE_CHOICE", 3),
        ("SHORT_ANSWER", 4),
        ("SESSION_REFLECTION", 5),
    ],
    "E": [
        ("ARTICLE_READING", 1),
        ("SUMMARY_READING", 2),
        ("SHORT_ANSWER", 3),
        ("SENTENCE_COMPLETION", 4),
        ("SESSION_REFLECTION", 5),
    ],
}

# 각 stepType의 순서를 전역으로 계산
GLOBAL_ORDER = {st.lower(): order for lvl in STEP_ORDER_MAP.values() for st, order in lvl}


def pretty(x):
    return json.dumps(x, ensure_ascii=False, indent=2)


def convert_quiz_package_to_csv():

    for json_file in PACKAGE_DIR.glob("*.json"):
        print(f"변환중 → {json_file}")

        output_path = OUTPUT_DIR / (json_file.stem + ".csv")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        rows = []
        all_columns = set()  # 실제 존재하는 컬럼만 생성

        courses = sorted(data.get("courses", []), key=lambda c: c.get("courseId", 0))

        for course in courses:
            session = course.get("sessions", [None])[0]
            if not session:
                continue

            # title이 절대 비어 있으면 안 됨
            title = course.get("courseName") or f"Course-{course.get('courseId')}"

            # 코스/세션 거의 최소 정보로만 저장
            course_text = pretty({
                "courseId": course.get("courseId"),
                "courseName": course.get("courseName"),
                "topic": course.get("topic"),
                "subTopic": course.get("subTopic"),
                "sessionId": session.get("sessionId"),
                "headline": session.get("headline"),
            })

            row = {
                "title": title,
                "course_text": course_text,
            }

            # 단계 저장
            step_by_level = {"N": {}, "I": {}, "E": {}}

            for quiz in session.get("quizzes", []):
                level = quiz.get("level")

                for step in quiz.get("steps", []):
                    step_type = step.get("contentType", "").lower()
                    contents = step.get("contents", [])

                    # flatten
                    flat = []
                    if isinstance(contents, list):
                        for it in contents:
                            if isinstance(it, list):
                                flat.extend(it)
                            else:
                                flat.append(it)
                    else:
                        flat = [contents]

                    step_by_level[level][step_type] = pretty(flat)

                    # 존재하는 컬럼만 기록 (중요)
                    colname = f"{level}_{step_type}"
                    all_columns.add(colname)

                    row[colname] = pretty(flat)

            rows.append(row)

        # ---- CSV 헤더 구성: title → course_text → 실제 존재한 컬럼만 정렬 -------
        # 컬럼 정렬 규칙:
        # level(N,I,E) → stepOrder → stepType
        def sort_key(col):
            if col in ["title", "course_text"]:
                return (-1, -1, col)

            lv, step_type = col.split("_", 1)
            return (["N", "I", "E"].index(lv), GLOBAL_ORDER.get(step_type, 999), step_type)

        fieldnames = ["title", "course_text"] + sorted(all_columns, key=sort_key)

        # ---- CSV 저장 ----
        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"✔ CSV 생성 완료 → {output_path}")


if __name__ == "__main__":
    convert_quiz_package_to_csv()