# === src/wrapper/course_wrapper.py ===
import json, logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# === 로거 설정 ===
logger = logging.getLogger(__name__)

def build_course_packages():
    """모든 토픽(economy, politics, society, world)에 대해 코스 + 세션 + 퀴즈 통합 JSON 생성"""
    today = datetime.now().strftime("%Y-%m-%d")

    BASE_DIR = Path(__file__).resolve().parents[2]
    COURSE_FILTERED_DIR = BASE_DIR / "data" / "course_db" / "filtered"
    QUIZ_DIR = BASE_DIR / "data" / "quiz"
    QUIZ_PACKAGE_DIR = QUIZ_DIR / "package"
    QUIZ_PACKAGE_DIR.mkdir(parents=True, exist_ok=True)

    # === 실행 ===
    topics = ["economy", "politics", "society", "world"]
    for topic in topics:
        try:
            course_path = COURSE_FILTERED_DIR / f"{topic}_{today}.json"
            output_path = QUIZ_PACKAGE_DIR / f"{topic}_{today}_package.json"

            if not course_path.exists():
                raise FileNotFoundError(f"코스 파일이 없습니다: {course_path}")

            # === 코스 파일 로드 ===
            with open(course_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                courses = data
            elif isinstance(data, dict):
                courses = data.get("courses", [])
            else:
                raise TypeError(f"예기치 못한 데이터 구조: {type(data)}")

            if not courses:
                logger.warning(f"[{topic}] course 데이터가 비어 있습니다.")
                continue

            # === 퀴즈 매핑 ===
            quiz_files = list(QUIZ_DIR.glob(f"{topic}_*_{today}.json"))
            if not quiz_files:
                logger.warning(f"[{topic}] 퀴즈 파일이 없습니다: {QUIZ_DIR}")

            quiz_map = defaultdict(list)

            for qf in quiz_files:
                parts = qf.stem.split("_")
                if len(parts) < 6:
                    logger.warning(f"[{topic}] 파일명 파싱 실패: {qf.name}")
                    continue

                cid = parts[1]
                sid = parts[2]
                contentType = "_".join(parts[3:-2])
                level = parts[-2]

                with open(qf, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    if len(data) > 0 and isinstance(data[0], dict):
                        quiz_entry = data[0]
                        contentType = quiz_entry.get("contentType", contentType)
                        level = quiz_entry.get("level", level)
                        contents = quiz_entry.get("contents", data)
                    else:
                        contents = data
                elif isinstance(data, dict):
                    contentType = data.get("contentType", contentType)
                    level = data.get("level", level)
                    contents = data.get("contents", data)
                else:
                    contents = []

                quiz_map[f"{cid}_{sid}"].append({
                    "contentType": contentType,
                    "level": level,
                    "sourceUrl": quiz_entry.get("sourceUrl", ""),
                    "contents": contents
                })

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

            # === 세션별 퀴즈 병합 ===
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
                        order = next(
                            (o for qname, o in STEP_ORDER_MAP.get(level, []) if qname == ctype),
                            None
                        )

                        if not order:
                            continue

                        if ctype == "SESSION_REFLECTION" and isinstance(q["contents"], dict):
                            level_map[level].append({
                                "stepOrder": order,
                                "contentType": ctype,
                                "contents": q["contents"]
                            })
                        else:
                            level_map[level].append({
                                "stepOrder": order,
                                "contentType": ctype,
                                "contents": q["contents"]
                            })

                    quizzes = []
                    for lvl in ["N", "I", "E"]:
                        if lvl in level_map:
                            quizzes.append({
                                "level": lvl,
                                "steps": sorted(level_map[lvl], key=lambda x: x["stepOrder"])
                            })
                    session["quizzes"] = quizzes

            # === 최종 저장 ===
            final_package = {"courses": courses}
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_package, f, ensure_ascii=False, indent=2)

            logger.info(f"[{topic}] 통합 패키지 생성 완료 → {output_path.resolve()}")

        except FileNotFoundError:
            logger.warning(f"[{topic}] 스킵 (코스 파일 없음)")
        except Exception as e:
            logger.error(f"[{topic}] 패키지 생성 중 오류 발생: {e}", exc_info=True)



if __name__ == "__main__":
    build_course_packages()