# === 표준 라이브러리 ===
import os
import json
import re
from datetime import datetime
from pathlib import Path
from collections import OrderedDict, defaultdict

# === 서드파티 라이브러리 ===
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from k_means_constrained import KMeansConstrained
from sentence_transformers import SentenceTransformer
import chromadb

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
    """sessionId → topic → subTopic 순서로 정렬"""
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

# === 메인 함수 ===
def generate_course_for_topic(topic: str):
    print(f"\n[{topic}] ChromaDB 불러오는 중...")

    # --- DB 경로 ---
    DB_DIR = BASE_DIR / "data" / "db" / topic
    chroma_client = chromadb.PersistentClient(path=str(DB_DIR))

    # --- 컬렉션 이름 감지 ---
    collections = chroma_client.list_collections()
    if not collections:
        print(f"[경고] {topic}: 컬렉션 없음. 스킵.")
        return

    try:
        first_col = collections[0]
        if hasattr(first_col, "name"):
            collection_name = first_col.name
        elif isinstance(first_col, dict):
            collection_name = first_col.get("name", f"{topic}_news")
        elif isinstance(first_col, str):
            collection_name = first_col
        else:
            collection_name = f"{topic}_news"
    except Exception:
        collection_name = f"{topic}_news"

    # --- 컬렉션 로드 (임베딩 함수 지정 필요 없음) ---
    collection = chroma_client.get_collection(name=collection_name)
    print(f"감지된 컬렉션: {collection_name}")

    # --- 데이터 + 임베딩 함께 불러오기 ---
    try:
        all_data = collection.get(include=["embeddings", "metadatas"], limit=5000)
        metadatas = [m for m in all_data.get("metadatas", []) if isinstance(m, dict)]
        embeddings = np.array(all_data.get("embeddings", []))
    except Exception as e:
        print(f"[오류] {topic} 데이터 로드 실패: {e}")
        return

    # --- 데이터 검증 ---
    if not metadatas or embeddings.size == 0:
        print(f"[{topic}] 문서 또는 임베딩이 비어 있음 → 스킵")
        return

    # --- 메타데이터 정리 ---
    docs = []
    for meta in metadatas:
        filtered_meta = {k: v for k, v in meta.items() if k != "deepsearchId"}
        docs.append(sort_session_keys(filtered_meta))

    print(f"[{topic}] 문서 {len(docs)}개 불러옴 / 임베딩 shape: {embeddings.shape}")

    # ===뉴스 요약문 기반 클러스터링 준비 ===
    valid_docs = [d for d in docs if isinstance(d.get("summary", ""), str) and len(d["summary"]) > 0]
    X = embeddings[:len(valid_docs)]

    if len(valid_docs) < 14:
        print(f"[{topic}] 데이터가 너무 적어 클러스터링 불가 → 스킵")
        return

    # === Balanced KMeans 클러스터링 ===
    n_clusters = 7
    clusterer = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=max(1, len(valid_docs) // n_clusters - 1),
        size_max=len(valid_docs) // n_clusters + 2,
        random_state=42
    )
    cluster_labels = clusterer.fit_predict(X)

    # === 클러스터 그룹화 ===
    grouped_by_cluster = defaultdict(list)
    for doc, label in zip(valid_docs, cluster_labels):
        grouped_by_cluster[label].append(doc)

    print(f"[{topic}] 클러스터 {len(grouped_by_cluster)}개 생성 완료")

    
    # === SentenceTransformer 로드 (1회만)
    embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")

    # === Hybrid 노이즈 필터 ===
    def filter_noise_hybrid(docs, threshold=0.5):
        """
        headline + summary 기반 노이즈 제거 (의미 일관성 중심)
        - headline만 쓸 때보다 주제 일관성이 높음
        """

        # 1️. 유효 텍스트 수 확인
        valid_texts = [
            f"{d.get('headline', '')} {d.get('summary', '')}".strip()
            for d in docs if d.get("headline") or d.get("summary")
        ]
        if len(valid_texts) <= 3:
            return docs  # 너무 적으면 필터링 생략

        # 2️. 임베딩 (hybrid 문장)
        embs = np.array([
            embedding_model.encode(t, convert_to_numpy=True, normalize_embeddings=True)
            for t in valid_texts
        ])

        # 3️. 중심 벡터 계산
        centroid = np.mean(embs, axis=0, keepdims=True)

        # 4️. 코사인 유사도 계산
        sims = cosine_similarity(embs, centroid).flatten()

        # 5️. 동적 threshold 적용 (데이터 편차 고려)
        mean_sim = np.mean(sims)
        std_sim = np.std(sims)
        adaptive_threshold = max(threshold, mean_sim - 0.4 * std_sim)

        # 6️. threshold 이상만 유지
        filtered_docs = [doc for doc, sim in zip(docs, sims) if sim >= adaptive_threshold]

        # 7️. 전부 제거되는 경우 대비
        return filtered_docs if filtered_docs else docs

        # === 여기서 실제 필터링 적용 ===
    for label in list(grouped_by_cluster.keys()):
        grouped_by_cluster[label] = filter_noise_hybrid(grouped_by_cluster[label])

    # === 최소 크기 기준으로 클러스터 정제 ===
    min_cluster_size = max(5, len(valid_docs) // n_clusters // 2)
    filtered_subtopics = {
        f"Cluster_{i+1}": v
        for i, v in grouped_by_cluster.items()
        if len(v) >= min_cluster_size
        }

    if not filtered_subtopics:
        print(f"[{topic}] 충분한 문서가 포함된 클러스터 없음 → 스킵")
        return

    # === 코스 생성 ===
    output = []

    for idx, (subTopic, group_news) in enumerate(filtered_subtopics.items(), start=1):
        for sid, news in enumerate(group_news, start=1):
            news["sessionId"] = sid
            group_news[sid - 1] = sort_session_keys(news)

        cleaned_sessions = []
        for s in group_news:
            filtered = {k: v for k, v in s.items() if k not in ("topic", "subTopic", "deepsearchId")}
            cleaned_sessions.append(sort_session_keys(filtered))

        headlines = [f"- {n.get('headline', '')}" for n in group_news]
        joined_headlines = "\n".join(headlines)

        TOPIC_SUBTOPICS = {
        "politics": "대통령실 OR 국회 OR 정당 OR 북한 OR 국방 OR 외교 OR 법률",
        "economy": "금융 OR 증권 OR 산업 OR 중소기업 OR 부동산 OR 물가 OR 무역",
        "society": "사건 OR 교육 OR 노동 OR 환경 OR 의료 OR 복지 OR 젠더",
        "world": "미국 OR 중국 OR 일본 OR 유럽 OR 중동 OR 아시아 OR 국제",
        "tech": "인공지능 OR 반도체 OR 로봇 OR 디지털 OR 과학기술 OR 연구개발 OR 혁신"
    }

        # --- 코스 설명 생성 ---
        prompt_course = f"""
출력은 반드시 JSON 형식으로 작성하세요.

당신은 뉴스 기반 학습 콘텐츠를 기획하는 에디터입니다.
아래 뉴스 요약들을 하나의 학습 코스로 묶어 코스명(courseName), 설명(courseDescription), 해시태그(sub_tag)를 만드세요.

---

### 목표
- 뉴스들의 공통 주제를 파악하고, 학습자가 인사이트를 얻을 수 있는 하나의 관점으로 묶으세요.
- 코스명은 "핵심 이슈 + 탐구 방향" 구조로 만드세요.
- 학문적이기보다는 신문·칼럼 스타일로 자연스럽고 세련된 문체를 유지하세요.

---

### 규칙
1. courseName
   - 형식: “핵심 이슈 : 탐구 방향/관점”  
     예: “부동산세 개편 : 정책과 불평등의 줄다리기”
   - 길이: 최소 20자, 25자 내외.
   - 끝맺음: 반드시 **명사형으로 끝나며 문장처럼 끝나지 않습니다.**
   - 톤: 문어체이되, 너무 학술적이지 않게. 신문 사설·칼럼처럼 자연스럽게.
   - 중복금지: 동일 토픽 내에서 이미 사용된 단어(‘시장’, ‘정책’, ‘금융’ 등)는 피하고, 독창적인 조합을 사용하세요.

2. courseDescription
   - 길이: 90~110자 내외.
   - 역할: 단순 요약이 아니라 “이 주제를 어떤 시선으로 탐구하는가”를 제시합니다.
]   - 반드시 **존댓말**로 작성해주세요.

3. subTopic
- courseName과 courseDescription을 바탕으로, 해당 코스의 핵심 주제를 대표하는 서브 토픽을 1개만 선택하세요.
- 아래 [서브 토픽 후보 목록]에서, 현재 topic에 해당하는 키워드 중 하나만 그대로 선택합니다.
- 새로운 단어나 조합을 만들지 마세요. 반드시 후보 중 하나를 그대로 사용하세요.

4. subTags
- 역할: 코스를 대표하는 고유 키워드 세트
- 형식: 2~3개의 해시태그, 최대 4글자 (예: "#정책갈등, #사회변화")
- 기준: 뉴스 헤드라인의 핵심 명사 기반,
- 일반어(‘경제’, ‘정치’) 대신 구체적·맥락형 표현 사용
- 중복금지: 동일 토픽 내 다른 코스와 겹치지 않게 생성

---

### 참고 포맷 가이드
| 유형 | 톤 | 예시 |
| --- | --- | --- |
| **① 사건형** | 중립적·분석적 | “세종시 이전 : 행정수도는 완성됐을까” |
| **② 인물형** | 내러티브·스토리텔링 | “오세훈과 재건축 : 정치와 부동산의 만남” |
| **③ 키워드형** | 담론적·철학적 | “‘공정’ : 세대 갈등의 정치학” |
| **④ 질문형** | 탐구적·호기심 유발 | “정치는 왜 갈등을 키우는가?” |
| **⑤ 대립형** | 논쟁적·균형적 | “집값 안정 vs 시장 자율 : 어디에 무게를 둘까” |
| **⑥ 감정형** | 서사적·공감적 | “분노의 청년세대 : 정치에 등을 돌리다” |

---

### 출력 형식(JSON)
{{
  "courseName": string,
  "courseDescription": string,
  "subTopic": string,
  "subTags": string
}}

[뉴스 제목 목록]
{joined_headlines}
[서브 토픽]
{TOPIC_SUBTOPICS.get(topic, "")}
"""

        try:
            resp_course = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt_course}],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            content = resp_course.choices[0].message.content
            if isinstance(content, str):
                meta_course = json.loads(content)
            elif isinstance(content, dict):
                meta_course = content
            else:
                raise ValueError("잘못된 JSON 응답 형식")
        except Exception as e:
            print(f"[{topic}] course 생성 실패: {e}")
            meta_course = {
                "courseName": f"{topic}_{subTopic}",
                "courseDescription": "자동 생성 실패 → 기본 설명",
                "subTopic": "",
                "subTags": ""
            }

        tag_candidates = []
        for k, v in meta_course.items():
            if "tag" in k.lower():
                if isinstance(v, list):
                    tag_candidates.extend(v)
                elif isinstance(v, str):
                    tag_candidates.append(v)

        clean_tags = sorted(set([t.strip() for t in tag_candidates if t.strip()]))
        subTags = []
        for tag_str in clean_tags:
            for token in tag_str.replace(",", " ").split():
                clean_token = token.strip()
                if clean_token and clean_token not in subTags:
                    subTags.append(clean_token)

        course_data = OrderedDict([
            ("courseId", idx),
            ("topic", topic),
            ("subTopic", meta_course.get("subTopic", subTopic)),
            ("subTags", subTags),
            ("courseName", meta_course.get("courseName", f"{topic}_{subTopic}")),
            ("courseDescription", meta_course.get("courseDescription", "")),
            ("sessions", cleaned_sessions),
        ])

        output.append(course_data)
        
    # === 저장 ===
    output_sorted = sorted(output, key=lambda x: x["courseId"])
    output_file = COURSE_DIR / f"{topic}_{today}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_sorted, f, ensure_ascii=False, indent=2, sort_keys=False)

    print(f"{topic} → 코스 파일 저장 완료: {output_file.resolve()}")