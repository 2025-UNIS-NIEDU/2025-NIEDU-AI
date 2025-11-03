# === 표준 라이브러리 ===
import os
import json
import re
from datetime import datetime
from pathlib import Path
from collections import OrderedDict, defaultdict

# === 서드파티 라이브러리 ===
import yaml
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from k_means_constrained import KMeansConstrained
from sentence_transformers import SentenceTransformer
import chromadb

def generate_all_courses():
    """뉴스 RAG 기반 전체 토픽(course) 자동 생성"""

    # === 날짜 ===
    today = datetime.now().strftime("%Y-%m-%d")

    # === 경로 설정 ===
    BASE_DIR = Path(__file__).resolve().parents[2]
    ENV_PATH = BASE_DIR / ".env"
    COURSE_DIR = BASE_DIR / "data" / "course_db"
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

    TOPIC_TRANSLATIONS = {
        "politics": "정치",
        "economy": "경제",
        "society": "사회",
        "world": "국제",
    }

    # === 메인 함수 ===
    def generate_course_for_topic(topic: str, embedding_model=None, client=None):
        if client is None:
            client = OpenAI(api_key=api_key)
        print(f"\n[{topic}] ChromaDB 불러오는 중...")

        # --- DB 경로 ---
        DB_DIR = BASE_DIR / "data" / "rag_db" / topic
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
        
        # --- 언론사 리스트 ---
        MAJOR_PUBLISHERS = [
            # --- 전국 종합지 ---
            "조선일보", "중앙일보", "동아일보",
            "한겨레", "경향신문", "한국일보",
            "서울신문", "문화일보", "국민일보", "세계일보",

            # --- 경제·비즈니스 ---
            "매일경제", "한국경제", "서울경제", "머니투데이",
            "아시아경제", "이데일리", "파이낸셜뉴스",
            "헤럴드경제", "디지털타임스", "비즈워치",

            # --- 방송사·통신사 ---
            "연합뉴스", "뉴스1", "뉴시스",
            "KBS", "MBC", "SBS", "YTN", "MBN", "JTBC",
            "채널A", "TV조선",

            # --- 온라인·탐사·오피니언 매체 ---
            "오마이뉴스", "프레시안", "미디어오늘",
            "더팩트", "뉴스타파", "노컷뉴스", "뉴스토마토",

            # --- IT·테크 전문 ---
            "아이뉴스24", "ZDNet코리아",

            # --- 지역 종합지 ---
            "부산일보", "국제신문", "매일신문", "영남일보",
            "광주일보", "전남일보", "전북일보",
            "강원일보", "강원도민일보", "경인일보", "인천일보",
            "울산매일", "경남도민일보", "제주일보", "한라일보"
        ]

        # --- 언론사 필터 적용 (메타데이터에서 publisher 기준으로 걸러내기) ---
        filtered_indices = [
            i for i, m in enumerate(metadatas)
            if m.get("publisher") in MAJOR_PUBLISHERS
        ]

        metadatas = [metadatas[i] for i in filtered_indices]
        embeddings = embeddings[filtered_indices]

        print(f"[{topic}] 언론사 필터 적용 후 문서 {len(metadatas)}개 남음")

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

        def clean_headline(text: str) -> str:
            """헤드라인 내 괄호·따옴표 제거 및 길이 제한"""
            text = re.sub(r"[\[\]\(\)\"\'“”‘’]", "", text)
            text = text.strip()
            return text[:60]

        for idx, (subTopic, group_news) in enumerate(filtered_subtopics.items(), start=1):
            # === 각 클러스터에서 상위 7개만 선택 ===
            group_news = group_news[:7]
            for sid, news in enumerate(group_news, start=1):
                news["sessionId"] = sid
                group_news[sid - 1] = sort_session_keys(news)

            cleaned_sessions = []
            for s in group_news:
                filtered = {k: v for k, v in s.items() if k not in ("topic", "subTopic", "deepsearchId")}
                cleaned_sessions.append(sort_session_keys(filtered))

            # === 헤드라인 간단 요약 ===
            headlines = [clean_headline(n.get("headline", "")) for n in group_news]
            joined_headlines = "; ".join(headlines)

            # === 토픽별 서브토픽 후보 ===
            TOPIC_SUBTOPICS = {
                "politics": "대통령실 OR 국회 OR 정당 OR 북한 OR 국방 OR 외교 OR 법률",
                "economy": "금융 OR 증권 OR 산업 OR 중소기업 OR 부동산 OR 물가 OR 무역",
                "society": "사건 OR 교육 OR 노동 OR 환경 OR 의료 OR 복지 OR 법률",
                "world": "미국 OR 중국 OR 일본 OR 유럽 OR 중동 OR 아시아 OR 국제",
            }

            max_headlines = 12  # 프롬프트에 넣을 헤드라인 수 제한
            joined_headlines = " / ".join([
                re.sub(r'["\'\[\]\(\):;]', '', n.get("headline", ""))[:60].strip()
                for n in group_news[:max_headlines]  # 상위 12개까지만 사용
                if n.get("headline")
            ])

            # --- YAML 프롬프트 로드 ---
            PROMPT_PATH = BASE_DIR / "src" / "course" / "prompt" / "course.yaml"
            with open(PROMPT_PATH, "r", encoding="utf-8") as f:
                prompt_conf = yaml.safe_load(f)

            prompt_course = f"""
    {prompt_conf['system_role']}

    {prompt_conf['rules']}

    {prompt_conf['examples']}

    {prompt_conf['output_format']}

    {prompt_conf['prompt_template'].format(
        joined_headlines=joined_headlines,
        subtopic_candidates=TOPIC_SUBTOPICS.get(topic, "")
    )}
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
                ("topic", TOPIC_TRANSLATIONS.get(topic, topic)),
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

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    import time

    embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    TOPICS = ["politics", "economy", "society", "world", "tech"]

    for topic in TOPICS:
        try:
            generate_course_for_topic(topic, embedding_model=embedding_model)
            time.sleep(1)  # DB 락 방지
        except Exception as e:
            print(f"[{topic}] 실행 중 오류: {e}")