# === src/wrapper/quiz_aggregator.py ===
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
QUIZ_DIR = BASE_DIR / "data" / "quiz"
AGGREGATED_DIR = QUIZ_DIR / "aggregated"
AGGREGATED_DIR.mkdir(parents=True, exist_ok=True)


def aggregate_quizzes_by_topic(topic: str):
    """
    data/quiz 내 topic별 퀴즈 JSON 파일들을 병합하여
    topic_quizzes.json 으로 저장

    ※ 파일명 예시:
      - economy_1_1_multi.json → courseId=1, sessionId=1
      - economy_3_2_short.json → courseId=3, sessionId=2
    """
    topic_files = list(QUIZ_DIR.glob(f"{topic}_*.json"))
    quizzes = {}

    for fpath in topic_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 파일 이름에서 courseId, sessionId 추출
            # 예: economy_2_1_multi.json → courseId=2, sessionId=1
            parts = fpath.stem.split("_")
            course_id = int(parts[1]) if len(parts) > 2 and parts[1].isdigit() else None
            session_id = int(parts[2]) if len(parts) > 3 and parts[2].isdigit() else None

            if course_id is None or session_id is None:
                continue

            key = f"{course_id}_{session_id}"
            quizzes.setdefault(key, {}).update(data)

        except Exception as e:
            print(f"[WARN] {fpath.name} 처리 중 오류: {e}")

    # === 저장 구조 ===
    # [
    #   {"courseId": 1, "sessionId": 1, "quizzes": {...}},
    #   {"courseId": 1, "sessionId": 2, "quizzes": {...}},
    # ]
    output_data = []
    for key, q in quizzes.items():
        try:
            cid, sid = map(int, key.split("_"))
            output_data.append({
                "courseId": cid,
                "sessionId": sid,
                "quizzes": q
            })
        except ValueError:
            continue

    output_path = AGGREGATED_DIR / f"{topic}_quizzes.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"[{topic}] 퀴즈 {len(topic_files)}개 병합 완료 → {output_path.resolve()}")
    return output_path