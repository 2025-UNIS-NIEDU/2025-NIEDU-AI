import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
COURSE_DIR = BASE_DIR / "data" / "course"

def select_session(tag: str = None, course_id: int = None, session_id: int = None):
    """
    [메인, #서브] 형태로 태그 세트를 선택하는 인터랙티브 세션 선택기
    - tag: 예) 'politics' 또는 '#대통령실' (선택 시 자동 필터링)
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
                all_courses.extend(data if isinstance(data, list) else [data])
        except Exception as e:
            print(f"[경고] {f.name} 읽기 실패: {e}")

    # === 코스별 태그 세트 추출 ===
    tag_sets = []
    for c in all_courses:
        tags = c.get("tags", [])
        if tags:
            tag_sets.append(tuple(tags))  # 튜플 형태로 저장 (중복 제거용)
    unique_tag_sets = sorted(set(tag_sets))

    # === 선택 가능한 태그 세트 출력 ===
    print("\n=== 선택 가능한 태그 세트 목록 ===")
    for i, tags in enumerate(unique_tag_sets, 1):
        formatted = ", ".join(tags)
        print(f"{i}. [{formatted}]")

    # === 세트 선택 ===
    while True:
        user_in = input("태그 세트 번호를 선택하세요: ").strip()
        if user_in.isdigit() and 1 <= int(user_in) <= len(unique_tag_sets):
            selected_tags = unique_tag_sets[int(user_in) - 1]
            break
        print("⚠️ 올바른 번호를 입력하세요 (예: 1, 2, 3...)")

    # === 해당 태그 세트와 일치하는 코스 필터링 ===
    filtered = [c for c in all_courses if tuple(c.get("tags", [])) == selected_tags]

    print(f"\n선택된 태그 세트: [{', '.join(selected_tags)}]")
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
    session["tags"] = course["tags"]

    print(f"\n 선택된 세션: {session['headline']}")
    return session


if __name__ == "__main__":
    # CLI 모드 실행
    session = select_session()
    print("\n=== 선택 결과 ===")
    print(json.dumps(session, ensure_ascii=False, indent=2))