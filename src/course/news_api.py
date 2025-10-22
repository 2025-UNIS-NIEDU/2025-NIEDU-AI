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
    """
    영어와 한자가 포함되지 않고,
    한글 비율이 일정 이상일 때만 True 반환.
    """
    if not text:
        return False

    # 영어 또는 한자 포함 시 바로 False
    if re.search(r"[A-Za-z]", text) or re.search(r"[\u3400-\u4DBF\u4E00-\u9FFF]", text):
        return False

    # 한글 비율 계산
    korean_chars = len(re.findall(r"[가-힣]", text))
    total_chars = len(text)

    return (korean_chars / total_chars) >= threshold if total_chars > 0 else False

# === DeepSearch 호출 함수 ===
def deepsearch_query(endpoint: str, subTopic: str, date_from: str, date_to: str,
                     page: int = 10, page_size: int = 70, order="published_at", direction="desc"):
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

# === deepsearchId, topic 고정 + 나머지 알파벳 순 ===
def sort_session_keys(session: dict) -> dict:
    """deepsearchId → topic 고정, 나머지는 알파벳 순으로 정렬"""
    if not isinstance(session, dict):
        return session

    ordered = []
    # 1️. deepsearchId, topic 먼저
    if "deepsearchId" in session:
        ordered.append(("deepsearchId", session["deepsearchId"]))
    if "topic" in session:
        ordered.append(("topic", session["topic"]))

    # 2️. 나머지 키 알파벳 순 정렬
    remaining = sorted(
        [(k, v) for k, v in session.items() if k not in ("deepsearchId", "topic")],
        key=lambda x: x[0].lower()
    )

    return dict(ordered + remaining)

# === 기사 수집 (다중 페이지 + 필터링) ===
def collect_articles_with_filter(topic: str, subTopic: str,
                                 date_from: str, date_to: str,
                                 min_length: int = 350, target_samples: int = 10,
                                 max_pages: int = 30):
    """
    토픽별 기사 수집 (길이 + 잘림 + 영어/한자 제외)
    각 subTopic을 최대 max_pages(기본 10페이지)까지 호출하여 수집
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
            summary = (a.get("summary") or "").strip()

            # 1️. 너무 짧은 경우 제외
            if len(summary) < min_length:
                continue

            # 2️. '...' 또는 '…'으로 끝나는 경우 제외 (요약 잘림)
            if summary.endswith("...") or summary.endswith("…"):
                continue

            # 3️. 영어·한자 포함 기사 제외
            if not is_purely_korean(summary):
                continue

            # 뉴스 데이터 정리
            session = {
                "deepsearchId": a.get("id"),
                "topic": topic,
                "subTopic": subTopic,
                "summary": summary,
                "contentUrl": a.get("content_url"),
                "headline": a.get("title"),
                "publishedAt": a.get("published_at"),
                "publisher": a.get("publisher"),
                "thumbnailUrl": a.get("thumbnail_url")
            }

            # deepsearchId/topic 먼저 정렬
            session = sort_session_keys(session)
            collected.append(session)

            # 🎯 목표 개수 도달 시 조기 종료
            if len(collected) >= target_samples:
                print(f"{topic}/{subTopic}: 목표 {target_samples}건 도달, 수집 중단")
                break

        # 페이지 간 딜레이
        time.sleep(1)

        # 수집 완료 시 루프 종료
        if len(collected) >= target_samples:
            break

    print(f"{topic}/{subTopic}: 총 {len(collected)}건 수집 완료 (필터링 후)")
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

    # "OR"로 구분된 subTopic 각각 쿼리로 분리
    subTopic_list = [s.strip() for s in re.split(r"\s*OR\s*", subTopics) if s.strip()]

    for sub in subTopic_list:
        print(f"--- 🔍 SubTopic 쿼리: {sub} ---")
        try:
            cleaned = collect_articles_with_filter(
                topic=topic,
                subTopic=sub,
                date_from=date_from,
                date_to=date_to,
                min_length=320,
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