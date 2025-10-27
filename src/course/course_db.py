import json
import sqlite3
from pathlib import Path

# === 경로 설정 ===
BASE_DIR = Path(__file__).resolve().parents[2]
COURSE_DB_DIR = BASE_DIR / "data" / "course_db"
COURSE_DB_DIR.mkdir(parents=True, exist_ok=True)

# === DB 저장 함수 ===
def save_course_to_db(topic: str, output: list):
    DB_PATH = COURSE_DB_DIR / f"{topic}.db"
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # === 테이블 생성 ===
    cur.execute("""
    CREATE TABLE IF NOT EXISTS courses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic TEXT,
        courseId INTEGER,
        courseName TEXT,
        courseDescription TEXT,
        subTopic TEXT,
        subTags TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        courseId INTEGER NOT NULL,
        sessionId INTEGER,
        headline TEXT,
        summary TEXT,
        publisher TEXT,
        publishedAt TEXT,
        thumbnailUrl TEXT,
        sourceUrl TEXT,
        FOREIGN KEY(courseId) REFERENCES courses(courseId)
    )
    """)

    # === 데이터 삽입 ===
    for course in output:
        cur.execute("""
            INSERT INTO courses (topic, courseId, courseName, courseDescription, subTopic, subTags)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            course.get("topic", topic),
            course.get("courseId"),
            course.get("courseName", ""),
            course.get("courseDescription", ""),
            course.get("subTopic", ""),
            ", ".join(course.get("subTags", []))
        ))

        for session in course.get("sessions", []):
            cur.execute("""
                INSERT INTO sessions (
                    courseId, sessionId, headline, summary,
                    publisher, publishedAt, thumbnailUrl, sourceUrl
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                course.get("courseId"),
                session.get("sessionId"),
                session.get("headline", ""),
                session.get("summary", ""),
                session.get("publisher", ""),
                session.get("publishedAt", ""),
                session.get("thumbnailUrl", ""),
                session.get("sourceUrl", "")
            ))

    conn.commit()
    conn.close()
    print(f"✅ {topic}.db 저장 완료 → {DB_PATH.resolve()}")


# === JSON → DB 변환 ===
json_files = sorted(COURSE_DB_DIR.glob("*.json"))
if not json_files:
    print("⚠️ course_db 폴더에 JSON 파일이 없습니다.")
else:
    for file_path in json_files:
        topic = file_path.stem.split("_")[0]  
        with open(file_path, "r", encoding="utf-8") as f:
            output = json.load(f)

        save_course_to_db(topic, output)