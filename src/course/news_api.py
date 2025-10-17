import os
import re
import requests
import json
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import time

# === 날짜 정의 ===
today = datetime.now().strftime("%Y-%m-%d")

# === 경로 설정 ===
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"
BACKUP_DIR = BASE_DIR / "data" / "backup"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# === 환경 변수 로드 ===
load_dotenv(dotenv_path=ENV_PATH, override=True)
DEEPSEARCH_API_KEY = os.getenv("DEEPSEARCH_API_KEY")

# === 한글 비율 계산 함수 ===
def is_mostly_korean(text: str, threshold: float = 0.3) -> bool:
    """한글 비율이 일정 이상이면 True"""
    if not text:
        return False
    korean_chars = len(re.findall(r"[가-힣]", text))
    total_chars = len(text)
    return (korean_chars / total_chars) >= threshold if total_chars > 0 else False

# === DeepSearch 호출 함수 ===
def deepsearch_query(endpoint: str, subTopic: str, date_from: str, date_to: str,
                     page: int = 1, page_size: int = 70, order="published_at", direction="desc"):
    """DeepSearch API 호출"""
    params = {
        "api_key": DEEPSEARCH_API_KEY,
        "q": subTopic,
        "page": page,
        "page_size": page_size,
        "date_from": date_from,
        "date_to": date_to,
        "order": order,
        "direction": direction
    }
    resp = requests.get(endpoint, params=params)
    resp.raise_for_status()
    return resp.json()

# === sessionId, topic 고정 + 나머지 알파벳 순 ===
def sort_session_keys(session: dict) -> dict:
    """sessionId → topic 고정, 나머지는 알파벳 순으로 정렬"""
    if not isinstance(session, dict):
        return session

    ordered = []
    # 1️. sessionId, topic 먼저
    if "sessionId" in session:
        ordered.append(("sessionId", session["sessionId"]))
    if "topic" in session:
        ordered.append(("topic", session["topic"]))

    # 2️. 나머지 키 알파벳 순 정렬
    remaining = sorted(
        [(k, v) for k, v in session.items() if k not in ("sessionId", "topic")],
        key=lambda x: x[0].lower()
    )

    return dict(ordered + remaining)

# === 기사 수집 (다중 페이지 + 필터링) ===
def collect_articles_with_filter(topic: str, subTopic: str,
                                 date_from: str, date_to: str,
                                 min_length: int = 300, target_samples: int = 100,
                                 max_pages: int = 5):
    """
    토픽별 기사 수집 (길이 + 잘림 + 영어 제외)
    """
    endpoint = f"https://api-v2.deepsearch.com/v1/articles/{topic}"
    collected = []
    page = 1

    while len(collected) < target_samples and page <= max_pages:
        resp = deepsearch_query(
            endpoint,
            subTopic=subTopic,
            date_from=date_from,
            date_to=date_to,
            page=page,
            page_size=100,
            order="published_at",
            direction="desc"
        )

        articles = resp.get("data", [])
        print(f"  ↳ {topic} | {page}페이지에서 {len(articles)}건 수신됨")

        for a in articles:
            content = (a.get("summary") or "").strip()

            # 1️. 내용이 너무 짧은 경우 제외
            if len(content) < min_length:
                continue

            # 2️. '...' 또는 '…'으로 끝나는 경우 제외 (요약 잘림)
            if content.endswith("...") or content.endswith("…"):
                continue

            # 3️. 영어 위주 기사 제외 (한글 비율 30% 미만)
            if not is_mostly_korean(content):
                continue

            session = {
                "sessionId": a.get("id"),
                "topic": a.get("sections")[0] if isinstance(a.get("sections"), list) else a.get("sections"),
                "author": a.get("author"),
                "content": content,
                "contentUrl": a.get("content_url"),
                "headline": a.get("title"),
                "publishedAt": a.get("published_at"),
                "publisher": a.get("publisher"),
                "thumbnailImage": a.get("thumbnail_url")
            }

            # 정렬 적용 (sessionId, topic → 나머지 ABC 순)
            session = sort_session_keys(session)
            collected.append(session)

            # 목표 개수 도달 시 조기 종료
            if len(collected) >= target_samples:
                break

        page += 1
        time.sleep(1)

    print(f"  {topic}: 총 {len(collected)}건 수집 완료 (필터링 후)")
    return collected

# === 토픽별 subTopic 정의 ===
TOPIC_SUBTOPICS = {
    "politics": "대통령실 OR 국회 OR 정당 OR 북한 OR 행정 OR 국방 OR 외교 OR 법률",
    "economy": "금융 OR 증권 OR 산업 OR 중소기업 OR 부동산",
    "society": "사건 OR 교육 OR 노동 OR 환경 OR 의료 OR 법률 OR 젠더",
    "world": "해외 OR 국제 OR 외신 OR 미국 OR 유럽 OR 중국 OR 일본 OR 중동 OR 아시아 OR 세계",
    "tech": "인공지능 OR 로봇 OR 반도체 OR 디지털 OR 우주 OR 과학기술 OR 연구개발 OR 혁신"
}

# === 실행 ===
date_from = "2025-10-16"
date_to = "2025-10-17"

for topic, subTopic in TOPIC_SUBTOPICS.items():
    print(f"\n=== [{topic}] 기사 수집 중... ===")
    try:
        cleaned = collect_articles_with_filter(
            topic,
            subTopic,
            date_from,
            date_to,
            min_length=300,
            target_samples=100,
            max_pages=5
        )

        backup_file = BACKUP_DIR / f"{topic}_{today}.json"
        with open(backup_file, "w", encoding="utf-8") as f:
            json.dump({"articles": cleaned}, f, ensure_ascii=False, indent=2, sort_keys=False)

        print(f"[정제 완료] {backup_file.resolve()} | {len(cleaned)}건 저장 완료")

    except Exception as e:
        print(f"[오류 발생] {topic}: {e}")

    time.sleep(2)