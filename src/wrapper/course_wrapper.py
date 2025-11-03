# === src/wrapper/course_wrapper.py ===
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def build_course_packages():
    """
    모든 토픽(economy, politics, society, world, tech)에 대해
    코스 + 세션 + 퀴즈 통합 JSON을 한 번에 생성
    """
    today = datetime.now().strftime("%Y-%m-%d")

    BASE_DIR = Path(__file__).resolve().parents[2]
    COURSE_FILTERED_DIR = BASE_DIR / "data" / "course_db" / "filtered"
    QUIZ_DIR = BASE_DIR / "data" / "quiz"
    QUIZ_PACKAGE_DIR = QUIZ_DIR / "package"
    QUIZ_PACKAGE_DIR.mkdir(parents=True, exist_ok=True)

    # 함수 이름 변경: build_course_package
    def build_course_package(topic: str):
        """
        1️. filtered/{topic}_{today}.json → 코스/세션 로드
        2️. quiz 폴더 스캔 → courseId + sessionId + level + contentType 매칭
        3️. 결과를 quiz/package/{topic}_{today}_package.json 으로 저장
        """
        course_path = COURSE_FILTERED_DIR / f"{topic}_{today}.json"
        output_path = QUIZ_PACKAGE_DIR / f"{topic}_{today}_package.json"

        if not course_path.exists():
            raise FileNotFoundError(f"코스 파일이 없습니다: {course_path}")

        with open(course_path, "r", encoding="utf-8") as f:
            courses = json.load(f)

        quiz_files = list(QUIZ_DIR.glob(f"{topic}_*.json"))
        if not quiz_files:
            print(f"퀴즈 파일이 없습니다: {QUIZ_DIR}")
        quiz_map = defaultdict(list)

        for qf in quiz_files:
            parts = qf.stem.split("_")
            if len(parts) < 6:
                print(f"파일명 파싱 실패: {qf.name}")
                continue

            cid = parts[1]
            sid = parts[2]
            level = parts[-2]
            contentType = "_".join(parts[3:-2])

            with open(qf, "r", encoding="utf-8") as f:
                data = json.load(f)

            contents = data.get("contents", data) if isinstance(data, dict) else data
            quiz_map[f"{cid}_{sid}"].append({
                "contentType": contentType,
                "level": level,
                "contents": contents
            })

        STEP_ORDER_MAP = {
            "N": [("SUMMARY_READING", 1), ("TERM_LEARNING", 2),
                  ("CURRENT_AFFAIRS", 3), ("OX_QUIZ", 4), ("MULTIPLE_CHOICE", 5)],
            "I": [("SUMMARY_READING", 1), ("MULTIPLE_CHOICE", 2),
                  ("SHORT_ANSWER", 3), ("SESSION_REFLECTION", 4)],
            "E": [("SUMMARY_READING", 1), ("SHORT_ANSWER", 2),
                  ("SENTENCE_COMPLETION", 3), ("SESSION_REFLECTION", 4)],
        }

        for course in courses:
            cid = str(course.get("courseId"))
            for session in course.get("sessions", []):
                sid = str(session.get("sessionId"))
                key = f"{cid}_{sid}"
                if key not in quiz_map:
                    continue

                level_map = defaultdict(list)
                for q in quiz_map[key]:
                    level = q["level"]
                    ctype = q["contentType"]
                    order = next((o for qname, o in STEP_ORDER_MAP.get(level, []) if qname == ctype), None)
                    if order:
                        level_map[level].append({
                            "stepOrder": order,
                            "contentType": ctype,
                            "contents": q["contents"]
                        })

                session["quizzes"] = [
                    {"level": lvl, "steps": sorted(steps, key=lambda x: x["stepOrder"])}
                    for lvl, steps in level_map.items()
                ]

        final_package = {"courses": courses}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_package, f, ensure_ascii=False, indent=2)

        print(f"[{topic}] 통합 패키지 생성 완료 → {output_path.resolve()}")
        return output_path


    # === 실제 실행 부분 ===
    topics = ["economy", "politics", "society", "world", "tech"]
    for topic in topics:
        try:
            build_course_package(topic)  
        except FileNotFoundError:
            print(f"{topic} 스킵 (코스 파일 없음)")


# === 메인 실행 ===
if __name__ == "__main__":
    build_course_packages()