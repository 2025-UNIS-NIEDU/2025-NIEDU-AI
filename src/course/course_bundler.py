import os
import json
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
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

# === session 정렬 함수 ===
def sort_session_keys(session: dict) -> dict:
    """sessionId → topic → subTopic → 나머지 알파벳순 정렬"""
    if not isinstance(session, dict):
        return session

    ordered = []
    for k in ("sessionId", "topic", "subTopic"):
        if k in session:
            ordered.append((k, session[k]))
    remaining = sorted(
        [(k, v) for k, v in session.items() if k not in ("sessionId", "topic", "subTopic")],
        key=lambda x: x[0].lower()
    )
    return OrderedDict(ordered + remaining)

# === 토픽 목록 ===
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

# === 메인 처리 ===
for topic in TOPIC_SUBTOPICS.keys():
    print(f"\n[{topic}] ChromaDB 불러오는 중...")

    DB_DIR = BASE_DIR / "data" / "db" / topic
    chroma_client = chromadb.PersistentClient(path=str(DB_DIR))

    # --- 컬렉션 감지 ---
    collections = chroma_client.list_collections()
    if not collections:
        print(f"[경고] {topic}: 컬렉션 없음. 스킵.")
        continue

    # v0.6.0 대응 (문자열 or dict)
    collection_name = (
        collections[0] if isinstance(collections[0], str)
        else collections[0].get("name", f"{topic}_news")
    )

    collection = chroma_client.get_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )
    print(f"📂 감지된 컬렉션: {collection_name}")

    # --- 데이터 불러오기 ---
    try:
        all_data = collection.get(include=["metadatas"], limit=5000)
        metadatas = all_data.get("metadatas", [])
    except Exception as e:
        print(f"[오류] {topic} 데이터 로드 실패: {e}")
        continue

    if not metadatas:
        print(f"{topic} — 불러온 문서 없음 (docs 비어있음)")
        continue

    docs = []
    for meta in metadatas:
        if not isinstance(meta, dict):
            continue
        filtered_meta = {k: v for k, v in meta.items() if k != "deepsearchId"}
        docs.append(sort_session_keys(filtered_meta))

    print(f"{topic} 뉴스 개수: {len(docs)}")

    # --- 샘플 미리보기 ---
    preview_docs = docs[:2]
    if preview_docs:
        print(f"샘플 문서 {len(preview_docs)}개 미리보기 ({topic}):")
        for d in preview_docs:
            print(f"  ├─ [{d.get('subTopic', '미지정')}] {d.get('headline', '')[:60]}...")
        print("-" * 80)

    # --- subTopic별 그룹화 ---
    grouped_by_sub = defaultdict(list)
    for d in docs:
        subTopic = d.get("subTopic", "기타") or "기타"
        grouped_by_sub[subTopic].append(d)

    output = []

    # --- 각 subTopic 단위로 코스 생성 ---
    for idx, (subTopic, group_news) in enumerate(grouped_by_sub.items(), start=1):
        # 세션 ID 부여 및 정렬
        for sid, news in enumerate(group_news, start=1):
            news["sessionId"] = sid
            group_news[sid - 1] = sort_session_keys(news)

        cleaned_sessions = []
        for s in group_news:
            filtered = {k: v for k, v in s.items() if k not in ("topic", "subTopic", "deepsearchId")}
            cleaned_sessions.append(sort_session_keys(filtered))    

        # 뉴스 요약 결합
        summaries = [
            f"- {n.get('headline', '')}: {n.get('summary', '')[:400]}"
            for n in group_news
        ]
        joined_summary = "\n".join(summaries)

        # --- 코스 설명 생성 ---
        prompt_course = f"""
출력은 반드시 JSON 형식으로 작성하세요.

당신은 학습형 뉴스 콘텐츠를 기획하는 작가이자 에디터입니다.
아래는 비슷한 주제의 뉴스 여러 개입니다.
이 뉴스들의 흐름을 바탕으로, **가독성이 높고 자연스러운 코스 제목(courseName)**과
짧지만 흐름이 보이는 **코스 설명(courseDescription)**을 함께 작성하세요.
코스 설명은 80-100자 내외

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

        # --- 태그 ---
        tags = [topic]
        if subTopic:
            tags.append(f"#{subTopic}")

        # --- 최종 코스 구성 ---
        course_data = OrderedDict([
            ("courseId", idx),
            ("tags", tags),
            ("courseName", meta_course.get("courseName", f"{topic}_{subTopic}")),
            ("courseDescription", meta_course.get("courseDescription", "")),
            ("sessions", cleaned_sessions),
        ])
        output.append(course_data)

    # --- 저장 ---
    output_sorted = sorted(output, key=lambda x: x["courseId"])
    output_file = COURSE_DIR / f"{topic}_{today}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_sorted, f, ensure_ascii=False, indent=2, sort_keys=False)

    print(f"💾 {topic} → 코스 파일 저장 완료: {output_file.resolve()}")

print("\n🎉 모든 토픽 코스 파일 생성 완료.")