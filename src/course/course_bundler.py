import os
import json
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
from collections import OrderedDict, defaultdict

# === 날짜 ===
today = datetime.now().strftime("%Y-%m-%d")

# === 경로 설정 ===
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"
COURSE_DIR = BASE_DIR / "data" / "course"
COURSE_DIR.mkdir(parents=True, exist_ok=True)

# === 환경 변수 로드 ===
load_dotenv(ENV_PATH, override=True)
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# === session 정렬 기준 함수 (RAG 동일)
def sort_session_keys(session: dict) -> dict:
    """sessionId(있을 경우) → deepsearchId → topic → 나머지 알파벳순 정렬"""
    if not isinstance(session, dict):
        return session

    ordered = []

    # 주요 키 순서
    if "sessionId" in session:
        ordered.append(("sessionId", session["sessionId"]))
    if "deepsearchId" in session:
        ordered.append(("deepsearchId", session["deepsearchId"]))
    if "topic" in session:
        ordered.append(("topic", session["topic"]))
    if "subTopic" in session:
        ordered.append(("subTopic", session["subTopic"]))

    # 나머지는 알파벳순
    remaining = sorted(
        [(k, v) for k, v in session.items() if k not in ("sessionId", "deepsearchId", "topic", "subTopic")],
        key=lambda x: x[0].lower()
    )

    return OrderedDict(ordered + remaining)


# === 토픽 정의 ===
TOPIC_SUBTOPICS = {
    "politics": "대통령실 OR 국회 OR 정당 OR 북한 OR 국방 OR 외교 OR 법률",
    "economy": "금융 OR 증권 OR 산업 OR 중소기업 OR 부동산 OR 물가 OR 무역",
    "society": "사건 OR 교육 OR 노동 OR 환경 OR 의료 OR 복지 OR 젠더",
    "world": "미국 OR 중국 OR 일본 OR 유럽 OR 중동 OR 아시아 OR 국제",
    "tech": "인공지능 OR 반도체 OR 로봇 OR 디지털 OR 과학기술 OR 연구개발 OR 혁신"
}

# === 임베딩 함수 ===
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api_key, model_name="text-embedding-3-small"
)

# === 처리 시작 ===
for topic, subtopic_query in TOPIC_SUBTOPICS.items():
    print(f"\n[{topic}] ChromaDB 불러오는 중...")

    DB_DIR = BASE_DIR / "data" / "db" / topic
    chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = chroma_client.get_or_create_collection(
        name=f"{topic}_news",
        embedding_function=embedding_fn
    )

    all_data = collection.get(include=["documents", "embeddings"])
    docs = []

    for doc in all_data["documents"]:
        try:
            parsed = json.loads(doc)
            docs.append(sort_session_keys(parsed))
        except Exception:
            continue

    if not docs:
        print(f"{topic} 데이터 없음, 스킵.")
        continue

    print(f"{topic} 뉴스 개수: {len(docs)}")

    # === subTopic별로 그룹화 ===
    grouped_by_sub = defaultdict(list)
    for d in docs:
        subTopic = d.get("subTopic", "기타")
        grouped_by_sub[subTopic].append(d)

    output = []

    # === 각 subTopic 단위로 코스 생성 ===
    for idx, (subTopic, group_news) in enumerate(grouped_by_sub.items(), start=1):
        # 세션 정렬 & ID 부여
        for sid, news in enumerate(group_news, start=1):
            news["sessionId"] = sid
            group_news[sid - 1] = sort_session_keys(news)

        # === 요약문 결합 ===
        summaries = [
            f"- {n.get('headline', '')}: {n.get('summary', '')[:400]}"
            for n in group_news
        ]
        joined_summary = "\n".join(summaries)

        # === 1️. Course Description 생성 ===
        prompt_course = f"""
출력은 반드시 JSON 형식으로 작성하세요.

당신은 학습형 뉴스 콘텐츠를 기획하는 작가이자 에디터입니다.
아래는 비슷한 주제의 뉴스 여러 개입니다.
이 뉴스들의 흐름을 바탕으로, **가독성이 높고 자연스러운 코스 제목(courseName)**과
짧지만 흐름이 보이는 **코스 요약(courseDescription)**을 함께 작성하세요.
출력은 반드시 JSON 형식으로 반환하세요.

🎯 작성 규칙
1. courseName:
   - 신문 헤드라인처럼 짧고 직관적이어야 합니다 (예: "AI가 바꾸는 산업 현장", "지방도시의 재도약, 관광이 해답이다").
   - 지나치게 기술적이거나 딱딱한 표현("이슈", "동향", "분석")은 피하세요.
   - 독자가 클릭하고 싶게 만드는 표현을 사용하세요.
   - 14~28자 이내로 작성하세요.

2. courseDescription:
   - 코스명 아래 들어갈 설명문으로, 뉴스 흐름을 2~3문장으로 간결하게 정리하세요.
   - 내용이 이어지는 이야기처럼 자연스럽게 써주세요.
   - "~을 이해한다" 대신 "~이 확산되고 있다", "~을 중심으로 논의가 이어진다" 같은 표현을 사용하세요.

출력 형식(JSON):
{{
  "courseName": string,
  "courseDescription": string
}}

뉴스 샘플 요약:
{joined_summary}
"""

        try:
            resp_course = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt_course}],
                response_format={"type": "json_object"}
            )
            meta_course = json.loads(resp_course.choices[0].message.content)
        except Exception as e:
            print(f"[{topic}] {subTopic} course 생성 실패: {e}")
            meta_course = {
                "courseName": f"{topic}_{subTopic}",
                "courseDescription": "자동 생성 실패 → 기본 설명"
            }

        # === 3️. Keyword 생성 (코스명에 실제 포함된 단어만) ===
        prompt_kw = f"""
당신은 SEO(검색엔진최적화) 전문가입니다.
아래의 코스명을 보고, **코스명에 실제로 등장하는 단어만** 사용하여
검색용 키워드를 3~5개 제시하세요.
출력은 반드시 JSON 형식으로 작성하세요.

🎯 규칙:
- 반드시 코스명에 직접 포함된 단어만 사용
- 코스명에 없는 단어는 절대 포함하지 말 것
- 단어는 모두 명사 형태
- 의미 중복 금지
- 조사, 동사, 형용사 제거

출력 형식:
{{"keywords": [string, string, string]}}

코스명: "{meta_course['courseName']}"
"""
        try:
            resp_kw = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt_kw}],
                response_format={"type": "json_object"}
            )
            meta_kw = json.loads(resp_kw.choices[0].message.content)
        except Exception as e:
            print(f"[{topic}] {subTopic} keyword 실패: {e}")
            meta_kw = {"keywords": []}

        # === 최종 course 데이터 구성 ===
        course_data = OrderedDict([
            ("courseId", idx),
            ("topic", topic),
            ("subTopic", subTopic),
            ("courseName", meta_course.get("courseName", f"{topic}_{subTopic}")),
            ("courseDescription", meta_course.get("courseDescription", "")),
            ("keywords", meta_kw.get("keywords", [])),
            ("sessions", group_news),
        ])
        output.append(course_data)

    # === 저장 ===
    output_sorted = sorted(output, key=lambda x: x["courseId"])
    output_file = COURSE_DIR / f"{topic}_{today}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_sorted, f, ensure_ascii=False, indent=2, sort_keys=False)

    print(f"{topic} → 코스 데이터 저장 완료: {output_file.resolve()}")

print("\n 모든 토픽 코스 파일 생성 완료.")