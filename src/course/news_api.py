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
def is_purely_korean(text: str, threshold: float = 0.3) -> bool:
    """영어나 한자 포함되지 않고 한글 비율이 일정 이상일 때 True"""
    if not text:
        return False

    if re.search(r"[A-Za-z]", text) or re.search(r"[\u3400-\u4DBF\u4E00-\u9FFF]", text):
        return False

    korean_chars = len(re.findall(r"[가-힣]", text))
    total_chars = len(text)
    return (korean_chars / total_chars) >= threshold if total_chars > 0 else False


# === DeepSearch API 호출 ===
def deepsearch_query(endpoint: str, subTopic: str, date_from: str, date_to: str,
                     page: int = 1, page_size: int = 70, order="published_at", direction="desc"):
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


# === key 정렬 ===
def sort_session_keys(session: dict) -> dict:
    """deepsearchId → topic → 나머지는 알파벳 순으로"""
    if not isinstance(session, dict):
        return session
    ordered = []
    for k in ("deepsearchId", "topic"):
        if k in session:
            ordered.append((k, session[k]))
    remaining = sorted(
        [(k, v) for k, v in session.items() if k not in ("deepsearchId", "topic")],
        key=lambda x: x[0].lower()
    )
    return dict(ordered + remaining)


# === 기사 수집 함수 ===
def collect_articles_with_filter(topic: str, subTopic: str,
                                 date_from: str, date_to: str,
                                 seen_ids: set,
                                 min_length: int = 250, target_samples: int = 10,
                                 max_pages: int = 50):
    """
    중복 제거 강화 버전
    전역 seen_ids를 이용해 중복 기사 완전 차단
    """
    endpoint = f"https://api-v2.deepsearch.com/v1/articles/{topic}"
    collected = []

    for page in range(1, max_pages + 1):
        print(f"  ↳ {topic}/{subTopic} | 페이지 {page} 호출 중...")

        try:
            resp = deepsearch_query(
                endpoint,
                subTopic=subTopic,
                date_from=date_from,
                date_to=date_to,
                page=page,
                page_size=target_samples,
                order="published_at",
                direction="desc"
            )
        except Exception as e:
            print(f"페이지 {page} 요청 실패: {e}")
            continue

        articles = resp.get("data", [])
        if not articles:
            print(f"페이지 {page}: 데이터 없음 → 중단")
            break

        for a in articles:
            article_id = a.get("id")

            # 전역 중복 필터링 (subTopic 간 중복 포함)
            if article_id in seen_ids:
                print(f"⚠️ 중복 기사 스킵: {a.get('title')}")
                continue
            seen_ids.add(article_id)

            summary = (a.get("summary") or "").strip()

            if len(summary) < min_length:
                continue
            if summary.endswith("...") or summary.endswith("…"):
                continue
            if not is_purely_korean(summary):
                continue

            session = {
                "deepsearchId": article_id,
                "topic": topic,
                "subTopic": subTopic,
                "summary": summary,
                "contentUrl": a.get("content_url"),
                "headline": a.get("title"),
                "publishedAt": a.get("published_at"),
                "publisher": a.get("publisher"),
                "thumbnailUrl": a.get("thumbnail_url")
            }

            session = sort_session_keys(session)
            collected.append(session)

            if len(collected) >= target_samples:
                print(f"{topic}/{subTopic}: 목표 {target_samples}건 도달, 수집 중단")
                break

        time.sleep(1)
        if len(collected) >= target_samples:
            break

    print(f"{topic}/{subTopic}: 총 {len(collected)}건 수집 완료 (중복 제거 후)")
    return collected


# === 토픽별 subTopic 정의 ===
TOPIC_SUBTOPICS = {
    "politics": "대통령실 OR 국회 OR 정당 OR 북한 OR 국방 OR 외교 OR 법률",
    "economy": "금융 OR 증권 OR 산업 OR 중소기업 OR 부동산 OR 물가 OR 무역",
    "society": "사건 OR 교육 OR 노동 OR 환경 OR 의료 OR 복지 OR 젠더",
    "world": "미국 OR 중국 OR 일본 OR 유럽 OR 중동 OR 아시아 OR 국제",
    "tech": "인공지능 OR 반도체 OR 로봇 OR 디지털 OR 과학기술 OR 연구개발 OR 혁신"
}


# === 실행 ===
date_from = "2025-10-20"
date_to = "2025-10-21"

for topic, subTopics in TOPIC_SUBTOPICS.items():
    print(f"\n=== [{topic}] 기사 수집 중... ===")
    collected_all = []
    seen_ids = set()  # 전역 중복 추적 세트

    subTopic_list = [s.strip() for s in re.split(r"\s*OR\s*", subTopics) if s.strip()]

    for sub in subTopic_list:
        print(f"--- 🔍 SubTopic 쿼리: {sub} ---")
        try:
            cleaned = collect_articles_with_filter(
                topic=topic,
                subTopic=sub,
                date_from=date_from,
                date_to=date_to,
                seen_ids=seen_ids, 
                min_length=250,
                target_samples=10
            )
            collected_all.extend(cleaned)
        except Exception as e:
            print(f"[오류 발생] {topic}/{sub}: {e}")
        time.sleep(2)

    backup_file = BACKUP_DIR / f"{topic}_{today}.json"
    with open(backup_file, "w", encoding="utf-8") as f:
        json.dump({"topic": topic, "articles": collected_all}, f, ensure_ascii=False, indent=2)

    print(f"[정제 완료] {backup_file.resolve()} | {len(collected_all)}건 저장 완료")