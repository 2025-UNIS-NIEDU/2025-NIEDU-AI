import json
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import os
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from collections import defaultdict
from openai import OpenAI

# === 절대경로 기반 프로젝트 루트 설정 ===
BASE_DIR = Path(__file__).resolve().parent.parent

# === 환경 변수 로드 ===
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === DB 경로 ===
DB_DIR = BASE_DIR / "data" / "db" / "politics"

# === Chroma Client ===
chroma_client = chromadb.PersistentClient(path=str(DB_DIR))

embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)

collection = chroma_client.get_or_create_collection(
    name="politics_news",
    embedding_function=embedding_fn
)

# === DB 전체 데이터 가져오기 ===
all_data = collection.get(include=["documents", "embeddings"])
docs, embeddings = [], []

for doc, emb in zip(all_data["documents"], all_data["embeddings"]):
    try:
        parsed = json.loads(doc)
        docs.append(parsed)
        embeddings.append(emb)
    except Exception as e:
        print("JSON 파싱 실패:", e)

X = np.array(embeddings)

# === KMeans 클러스터링 ===
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
labels = kmeans.fit_predict(X)

# === 클러스터별 묶기 ===
clusters = defaultdict(list)
for doc, label in zip(docs, labels):
    clusters[int(label)].append(doc)

# === OpenAI Client ===
client = OpenAI(api_key=OPENAI_API_KEY)

# === 미리 정의된 세부 키워드 ===
defined_keywords = ["대통령실", "국회", "정당", "북한", "행정", "국방", "외교"]

output = []

# === PREFIX (섹션 구분용) ===
SECTION_PREFIX = "P"  # politics → P

for idx, (cid, group) in enumerate(clusters.items(), start=1):
    reduced_info = [
        {"title": g.get("title", ""), "summary": g.get("summary", "")}
        for g in group
    ]

    # === LLM 프롬프트 ===
    prompt = f"""
아래 기사들의 title과 summary를 참고하여, 하나의 코스 메타데이터를 생성하라.

출력은 JSON 하나만 한다.
출력 구조:
{{
  "course_id": string,                      # 예: "P01", "P02" (정치 도메인 내 고유 식별자)
  "course_name": string,                    # 주제를 대표하는 짧은 이름 (약 5~10자)
  "description": string,                    # 단순 요약이 아닌, 기사들의 흐름이 느껴지는 스토리라인 문장 (약 40~70자)
  "keywords": [string, string, string],     # 아래 목록 중 이 코스와 가장 관련 깊은 주제 2개 이상
  "progress": int                           # User 사용 로그 반영 (1~100 사이 퍼센트 단위 숫자)
}}

선택 가능한 주제 목록:
{defined_keywords}

규칙:
- description은 간결하고 자연스러운 한 문장으로 작성한다.
- keywords는 반드시 위 목록 중 1개 이상을 선택한다.
- 새로운 단어나 조합을 만들지 않는다.
- course_name은 핵심 주제를 짧게 표현한다.
- course_id는 "P" + 2자리 숫자로 설정한다. 예: "P01", "P02"

articles (title + summary):
{json.dumps(reduced_info, ensure_ascii=False, indent=2)}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        course_meta = json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"[클러스터 {cid}] LLM JSON 생성 실패:", e)
        course_meta = {
            "course_name": f"코스 {idx}",
            "description": "자동 생성 실패 → 기본 설명",
            "keywords": []
        }

    # === course_id 강제 지정 (문자+숫자 조합) ===
    course_meta["course_id"] = f"{SECTION_PREFIX}{idx:02d}"

    # === progress 초기화 ===
    course_meta["progress"] = 0

    # === articles 추가 ===
    course_meta["articles"] = group
    output.append(course_meta)

# === JSON 저장 ===
output_file = BASE_DIR / "data" / "course" / "politics_clustered.json"
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n 클러스터링 + LLM 후처리 완료 → {output_file.resolve()}")
