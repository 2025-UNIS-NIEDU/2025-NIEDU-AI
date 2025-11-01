import json
import csv
from pathlib import Path

# === 경로 설정 ===
BASE_DIR = Path(__file__).resolve().parents[2]
JSON_PATH = BASE_DIR / "data" / "course_db" / "economy_2025-11-01.json"
CSV_PATH = BASE_DIR / "data" / "course_db" / "economy_2025-11-01.csv"

# === JSON 로드 ===
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# === CSV 헤더 정의 ===
headers = ["courseId", "courseName", "topic", "subTopic", "sessions_summary"]

# === CSV 변환 ===
rows = []
for course in data:
    sessions = course.get("sessions", [])
    # 각 세션의 "헤드라인 - 언론사" 조합을 한 줄로 묶기
    session_texts = [
        f"{s.get('headline', '').strip()} ({s.get('publisher', '').strip()})"
        for s in sessions
        if s.get("headline")
    ]
    # 여러 세션을 하나의 셀 안에 줄바꿈으로 연결
    sessions_summary = "\n".join(session_texts)

    rows.append({
        "courseId": course.get("courseId"),
        "courseName": course.get("courseName"),
        "topic": course.get("topic"),
        "subTopic": course.get("subTopic"),
        "sessions_summary": sessions_summary
    })

# === CSV 저장 ===
with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ 변환 완료! → {CSV_PATH.name}")
