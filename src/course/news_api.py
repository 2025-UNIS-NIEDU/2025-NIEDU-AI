import os
import requests
import json
from pathlib import Path
from dotenv import load_dotenv

# === .env 불러오기 ===
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

DEEPSEARCH_API_KEY = os.getenv("DEEPSEARCH_API_KEY")

DEEPSEARCH_ENDPOINT = "https://api-v2.deepsearch.com/v1/articles/politics"

BASE_DIR = Path(__file__).resolve().parent.parent
BACKUP_DIR = BASE_DIR / "data" / "backup"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# === DeepSearch 호출 함수 ===
def deepsearch_query(keyword: str, date_from: str, date_to: str, page_size: int = 70, order="published_at", direction="asc"):
    params = {
        "api_key": DEEPSEARCH_API_KEY,
        "q": keyword,
        "page": 1,
        "page_size": page_size,
        "date_from": date_from,
        "date_to": date_to,
        "order": order,
        "direction": direction
    }
    resp = requests.get(DEEPSEARCH_ENDPOINT, params=params)
    resp.raise_for_status()
    return resp.json()

# === 기사 수집 (고정 기간) ===
def collect_articles_fixed_period(keyword: str, date_from: str, date_to: str, samples: int = 70):
    resp = deepsearch_query(
        keyword,
        date_from=date_from,
        date_to=date_to,
        page_size=samples,
        order="published_at",
        direction="asc"
    )
    results = resp.get("data", [])
    return results[:samples]

# === 필요한 필드만 정제 ===
def clean_articles(articles):
    cleaned = []
    for a in articles:
        cleaned.append({
            "news_id": a.get("id"),
            "sections": a.get("sections", []),
            "title": a.get("title"),
            "publisher": a.get("publisher"),
            "author": a.get("author"),
            "summary": a.get("summary"),
            "thumbnail_url": a.get("thumbnail_url"),
            "content_url": a.get("content_url"),
            "published_at": a.get("published_at")
        })
    return cleaned

# === 실행 ===
keyword = "대통령실 OR 국회 OR 정당 OR 북한 OR 행정 OR 국방 OR 외교"
date_from = "2025-10-01"
date_to = "2025-10-09"

all_articles = collect_articles_fixed_period(keyword, date_from, date_to, samples=70)

# === 필드 정제 ===
cleaned_articles = clean_articles(all_articles)

backup_file = BACKUP_DIR / "politics.json"

with open(backup_file, "w", encoding="utf-8") as f:
    json.dump({"articles": cleaned_articles}, f, ensure_ascii=False, indent=2)

print(f"[정제 완료] {backup_file.resolve()} | 기간: {date_from} ~ {date_to} | 총 {len(cleaned_articles)}건 저장")