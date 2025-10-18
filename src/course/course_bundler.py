import os
import json
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from openai import OpenAI
from datetime import datetime
from collections import OrderedDict
from collections import Counter

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

    # 1️⃣ 주요 키 순서대로 추가 (존재할 경우만)
    if "sessionId" in session:
        ordered.append(("sessionId", session["sessionId"]))
    if "deepsearchId" in session:
        ordered.append(("deepsearchId", session["deepsearchId"]))
    if "topic" in session:
        ordered.append(("topic", session["topic"]))

    # 2️⃣ 나머지 키는 알파벳 순으로 정렬
    remaining = sorted(
        [(k, v) for k, v in session.items() if k not in ("sessionId", "deepsearchId", "topic")],
        key=lambda x: x[0].lower()
    )

    return OrderedDict(ordered + remaining)

# === 토픽 정의 ===
TOPIC_SUBTOPICS = {
    "politics": "대통령실 OR 국회 OR 정당 OR 북한 OR 행정 OR 국방 OR 외교 OR 법률",
    "economy": "금융 OR 증권 OR 산업 OR 중소기업 OR 부동산",
    "society": "사건 OR 교육 OR 노동 OR 환경 OR 의료 OR 법률 OR 젠더",
    "world": "해외 OR 국제 OR 외신 OR 미국 OR 유럽 OR 중국 OR 일본 OR 중동 OR 아시아 OR 세계",
    "tech": "인공지능 OR 로봇 OR 반도체 OR 디지털 OR 우주 OR 과학기술 OR 연구개발 OR 혁신"
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
    docs, embeddings = [], []

    for doc, emb in zip(all_data["documents"], all_data["embeddings"]):
        try:
            parsed = json.loads(doc)
            docs.append(sort_session_keys(parsed))  # RAG 정렬 재적용
            embeddings.append(emb)
        except Exception:
            continue

    if not docs:
        print(f"{topic} 데이터 없음, 스킵.")
        continue

    X = np.array(embeddings)
    print(f"{topic} 뉴스 개수: {len(docs)}")

    # === 1. KMeans 기본 실행 ===
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)

    # === 2. 클러스터 크기 분포 출력 ===
    counts = Counter(labels)
    print(f"[{topic}] 초기 클러스터 분포:", dict(counts))

    # === 각 클러스터별 세션 수 출력
    from collections import Counter
    counts = Counter(labels)
    print(f"\n[{topic}] 클러스터별 세션 수 분포:")
    for cid, cnt in sorted(counts.items()):
        print(f"  🟢 클러스터 {cid}: {cnt}개 세션")

    # === 결과 리스트 초기화 ===
    output = []

    # === 각 코스(cluster) 단위 처리 ===
    for cluster_id in range(n_clusters):
        cluster_news = [docs[i] for i, label in enumerate(labels) if label == cluster_id]
        if not cluster_news:
            continue

        # 세션 ID 숫자 부여 (1부터 시작)
        for idx, news in enumerate(cluster_news, start=1):
            news["sessionId"] = idx
            cluster_news[idx - 1] = sort_session_keys(news)
            
        # === 세션 요약문 ===
        summaries = [
            f"- {news.get('headline', '')}: {news.get('content', '')[:150]}"
            for news in cluster_news[:7]
        ]
        joined_summary = "\n".join(summaries)

        # === 1️. Course Description 생성 ===
        prompt_course = f"""
출력은 반드시 JSON 형식으로 작성하세요.

당신은 학습형 뉴스 콘텐츠를 기획하는 작가이자 에디터입니다.
아래는 비슷한 주제의 뉴스 여러 개입니다.
이 뉴스들의 흐름을 바탕으로, **가독성이 높고 자연스러운 코스 제목(courseName)**과
짧지만 흐름이 보이는 **코스 요약(courseDescription)**을 함께 작성하세요.

🎯 작성 규칙
1. courseName:
   - 신문 헤드라인처럼 짧고 직관적이어야 합니다 (예: "AI가 바꾸는 산업 현장", "지방도시의 재도약, 관광이 해답이다").
   - 지나치게 기술적이거나 딱딱한 표현("이슈", "동향", "분석")은 피하세요.
   - 독자가 클릭하고 싶게 만드는 표현을 사용하세요.
   - 15~25자 이내로 작성하세요.

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
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt_course}],
                response_format={"type": "json_object"}
            )
            meta_course = json.loads(resp_course.choices[0].message.content)
        except Exception as e:
            print(f"[{topic} 클러스터 {cluster_id}] 코스 생성 실패: {e}")
            meta_course = {
                "courseName": f"{topic} 이슈 {cluster_id+1}",
                "courseDescription": "자동 생성 실패 → 기본 설명"
            }

        # === 2️. SubTopic 생성 (사전 정의된 후보만 선택) ===
        defined_subtopics = TOPIC_SUBTOPICS[topic].split(" OR ")
        session_texts = "\n".join([n.get("content", "")[:200] for n in cluster_news])

        prompt_subtopic = f"""
출력은 반드시 JSON 형식으로 작성하세요.

당신은 뉴스 주제 분류 전문가입니다.
아래는 여러 개의 뉴스 기사 내용입니다.
이 뉴스들은 모두 '{topic}' 분야에 속하며,
아래에 제시된 세부 주제 목록 중에서 이 클러스터의 뉴스들이 가장 밀접한 항목 2~4개를 선택하세요.

🎯 선택 가능한 세부 주제:
{defined_subtopics}

출력 형식:
{{"subTopic": [string, string, string]}}

규칙:
- 반드시 위 목록에서만 선택할 것 (새로운 단어 추가 금지)
- 중복 없이 중요도 순서대로 정렬

뉴스 요약 (참고용):
{session_texts}
"""
        try:
            resp_sub = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt_subtopic}],
                response_format={"type": "json_object"}
            )
            meta_sub = json.loads(resp_sub.choices[0].message.content)
        except Exception as e:
            print(f"[{topic} 클러스터 {cluster_id}] subTopic 실패: {e}")
            meta_sub = {"subTopic": [topic]}

        # === 3️. Keyword 생성 (코스명에 실제 포함된 단어만) ===
        prompt_kw = f"""
출력은 반드시 JSON 형식으로 작성하세요.

당신은 SEO(검색엔진최적화) 전문가입니다.
아래의 코스명을 보고, **코스명에 실제로 등장하는 단어만** 사용하여
검색용 키워드를 3~5개 제시하세요.

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
            print(f"[{topic} 클러스터 {cluster_id}] keyword 실패: {e}")
            meta_kw = {"keywords": []}

        # === 4️. 최종 course 데이터 구성 (정렬 고정)
        course_data = OrderedDict([
            ("courseId", cluster_id + 1),
            ("courseName", meta_course.get("courseName", f"{topic} {cluster_id+1}")),
            ("courseDescription", meta_course.get("courseDescription", "")),
            ("subTopic", meta_sub.get("subTopic", [topic])),
            ("keywords", meta_kw.get("keywords", [])),
            ("sessions", cluster_news),  # RAG 정렬 유지
        ])
        output.append(course_data)

    # === 5️. output 정렬: courseId 기준
    output_sorted = sorted(output, key=lambda x: x["courseId"])

    # === 6️. JSON 저장 (순서 유지)
    output_file = COURSE_DIR / f"{topic}_{today}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_sorted, f, ensure_ascii=False, indent=2, sort_keys=False)

    print(f"{topic} → 코스 데이터 저장 완료: {output_file.resolve()}")

print("\n 모든 토픽 코스 파일 생성 완료.")