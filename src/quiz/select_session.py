import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
COURSE_DIR = BASE_DIR / "data" / "course_db" / "filtered"

TOPIC_TRANSLATION = {
    "정치": "politics",
    "경제": "economy",
    "사회": "society",
    "국제": "world",
}

def select_session():
    """입력 없이 모든 코스·세션을 자동 순회"""
    files = sorted(COURSE_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError("data/course_db/filtered 폴더에 코스 파일이 없습니다.")

    all_sessions = []

    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception as e:
            print(f"[경고] {f.name} 읽기 실패: {e}")
            continue

        # --- 구조 보정 ---
        if isinstance(data, dict) and "courses" in data:
            courses = data["courses"]
        elif isinstance(data, list):
            courses = data
        else:
            print(f"[경고] {f.name}의 구조를 인식할 수 없습니다.")
            continue

        for course in courses:
            topic_kor = course.get("topic", "미지정")
            topic_eng = TOPIC_TRANSLATION.get(topic_kor, topic_kor)
            course_id = course.get("courseId")
            course_name = course.get("courseName", "무제")
            sub_topic = course.get("subTopic", {})
            sub_tags = course.get("subTags", [])
            sessions = course.get("sessions", [])

            for s in sessions:
                s["courseId"] = course_id
                s["courseName"] = course_name
                s["topic"] = topic_eng
                s["subTopic"] = sub_topic
                s["subTags"] = sub_tags
                all_sessions.append(s)

    print(f"총 세션 수: {len(all_sessions)}개")
    return all_sessions


if __name__ == "__main__":
    sessions = select_session()
    print(json.dumps(sessions[:5], ensure_ascii=False, indent=2))  