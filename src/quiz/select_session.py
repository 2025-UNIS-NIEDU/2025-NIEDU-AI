import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
COURSE_DIR = BASE_DIR / "data" / "course_db" / "filtered"

# === 한글 → 영어 토픽 변환 맵 ===
TOPIC_TRANSLATION = {
    "정치": "politics",
    "경제": "economy",
    "사회": "society",
    "국제": "world",
}

def select_session(topic: str = None, sub_topic: str = None, course_id: int = None, session_id: int = None):
    """
    [topic / subTopics] 기반 세션 선택기
    - topic: 예) 'economy', 'politics'
    - sub_topic: 예) '#산업혁신', '#기술트렌드'
    """
    # === 코스 파일 탐색 ===
    files = list(COURSE_DIR.glob(f"*.json"))
    if not files:
        raise FileNotFoundError("data/course 폴더에 코스 파일이 없습니다.")

    # === 모든 코스 로드 ===
    all_courses = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)

                if isinstance(data, dict) and "courses" in data:
                    all_courses.extend(data["courses"])
                elif isinstance(data, list):
                    all_courses.extend(data)
                else:
                    print(f"[경고] {f.name}의 데이터 구조를 인식할 수 없습니다.")
        except Exception as e:
            print(f"[경고] {f.name} 읽기 실패: {e}")

    unique_courses = {}
    for c in all_courses:
        key = f"{c.get('topic', '')}_{c.get('courseId', '')}"
        unique_courses[key] = c
    all_courses = list(unique_courses.values())

    # === topic/subTopics 목록 추출 ===
    topic_map = {}
    for c in all_courses:
        t = c.get("topic", "미지정")
        subs = c.get("subTopics", [])
        topic_map.setdefault(t, set()).update(subs)

    # === topic 선택 ===
    print("\n=== 선택 가능한 토픽 목록 ===")
    for i, t in enumerate(sorted(topic_map.keys()), 1):
        print(f"{i}. {t}")

    if topic is None:
        while True:
            user_in = input("토픽 번호를 선택하세요: ").strip()
            if user_in.isdigit() and 1 <= int(user_in) <= len(topic_map):
                topic = sorted(topic_map.keys())[int(user_in) - 1]
                break
            print(" 올바른 번호를 입력하세요 (예: 1, 2, 3...) ")

    # 영어 변환 
    topic_eng = TOPIC_TRANSLATION.get(topic, topic)

    # === 서브토픽 선택 ===
    subtopics = sorted(topic_map.get(topic, []))
    print(f"\n[{topic}] 관련 서브토픽:")
    for i, s in enumerate(subtopics, 1):
        print(f"{i}. {s}")

    if sub_topic is None and subtopics:
        while True:
            user_in = input("서브토픽 번호를 선택하세요 (없으면 Enter): ").strip()
            if not user_in:
                sub_topic = None
                break
            if user_in.isdigit() and 1 <= int(user_in) <= len(subtopics):
                sub_topic = subtopics[int(user_in) - 1]
                break
            print(" 올바른 번호를 입력하세요 (예: 1, 2, 3...) ")

    # === 해당 조건과 일치하는 코스 필터 ===
    filtered = []
    for c in all_courses:
        if c.get("topic") != topic:
            continue
        if sub_topic and sub_topic not in c.get("subTopics", []):
            continue
        filtered.append(c)

    if not filtered:
        raise ValueError(f"'{topic}' 관련 코스를 찾을 수 없습니다.")

    print(f"\n선택된 토픽: {topic} ({topic_eng})")
    if sub_topic:
        print(f"선택된 서브토픽: {sub_topic}")
    print("=== 코스 목록 ===")
    for c in filtered:
        print(f"- {c['courseId']}: {c['courseName']}")

    # === 코스 선택 ===
    if course_id is None:
        while True:
            try:
                course_id = int(input("\n선택할 courseId 입력: "))
                break
            except ValueError:
                print("숫자만 입력하세요.")

    course = next((c for c in filtered if c["courseId"] == course_id), None)
    if not course:
        raise ValueError("해당 courseId를 찾을 수 없습니다.")

    # === 세션 선택 ===
    if session_id is None:
        print(f"\n[{course['courseName']}] 세션 목록:")
        for i, s in enumerate(course["sessions"], start=1):
            print(f"{i}. {s['headline']}")
        while True:
            try:
                session_id = int(input("\n선택할 세션 번호 입력: "))
                break
            except ValueError:
                print("숫자만 입력하세요.")

    session = next((s for s in course["sessions"] if s["sessionId"] == session_id), None)
    if not session:
        raise ValueError("해당 세션을 찾을 수 없습니다.")

    # === 메타데이터 추가 ===
    session["courseId"] = course["courseId"]
    session["courseName"] = course["courseName"]
    session["topic"] = topic_eng  # 영어 토픽으로 저장
    session["subTopic"] = course.get("subTopic", {})
    session["subTags"] = course.get("subTags", [])

    print(f"\n선택된 세션: {session['headline']}")
    return session


if __name__ == "__main__":
    # CLI 모드 실행
    session = select_session()
    print("\n=== 선택 결과 ===")
    print(json.dumps(session, ensure_ascii=False, indent=2))