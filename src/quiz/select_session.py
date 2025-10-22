import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
COURSE_DIR = BASE_DIR / "data" / "course"

def select_session(topic=None):
    if topic is None:
        topics = ["politics", "economy", "society", "world", "tech"]
        print("\n=== 선택 가능한 토픽 ===")
        for i, t in enumerate(topics, 1):
            print(f"{i}. {t}")
        topic = topics[int(input("토픽 번호를 선택하세요: ")) - 1]

    target_file = next((f for f in COURSE_DIR.glob(f"{topic}_*.json")), None)
    if not target_file:
        raise FileNotFoundError(f"{topic} 관련 코스 파일이 없습니다.")

    with open(target_file, "r", encoding="utf-8") as f:
        courses = json.load(f)
        if not isinstance(courses, list):
            courses = [courses]

    print(f"\n불러온 파일: {target_file.name}")
    print("\n=== 코스 목록 ===")
    for c in courses:
        print(f"- {c['courseId']}: {c['courseName']}")

    target_id = input("\n원하는 courseId를 입력하세요: ").strip()
    course = next((c for c in courses if str(c["courseId"]) == target_id), None)
    if not course:
        raise ValueError("해당 courseId를 찾을 수 없습니다.")

    print(f"\n[{course['courseName']}] 세션 목록:")
    for i, s in enumerate(course["sessions"], start=1):
        print(f"{i}. {s['headline']}")

    session = course["sessions"][int(input("\n선택할 세션 번호를 입력하세요: ")) - 1]
    session["courseId"] = course["courseId"]
    session["courseName"] = course["courseName"]
    print(f"\n 선택된 세션: {session['headline']}")
    return session

if __name__ == "__main__":
    session = select_session()
    print("\n=== 선택 결과 ===")
    print(json.dumps(session, ensure_ascii=False, indent=2))