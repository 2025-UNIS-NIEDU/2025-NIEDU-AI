import os
import json
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# === [1] 프로젝트 루트 경로 설정 ===
# 현재 파일의 상위 폴더(예: scripts → project_root)
BASE_DIR = Path(__file__).resolve().parent.parent

# === [2] 환경 변수 로드 ===
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === [3] 경로 설정 ===
BACKUP_FILE = BASE_DIR / "data" / "backup" / "politics.json"
DB_DIR = BASE_DIR / "data" / "db" / "politics"

# === [4] 폴더 생성 === 
DB_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_FILE.parent.mkdir(parents=True, exist_ok=True)

# === Chroma Client 설정 ===
client = chromadb.PersistentClient(path=str(DB_DIR))
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)
collection = client.get_or_create_collection(
    name="politics_news",
    embedding_function=embedding_fn
)

# === JSON 로드 ===
with open(BACKUP_FILE, "r", encoding="utf-8") as f:
    raw = json.load(f)

# DeepSearch JSON 구조에 맞게 처리
articles = raw["data"] if "data" in raw else raw.get("articles", [])

# === DB 입력 데이터 준비 ===
docs, metas, ids = [], [], []

for i, item in enumerate(articles):
    # document: JSON 전체를 문자열화
    docs.append(json.dumps(item, ensure_ascii=False))

    # metadata: 단순 문자열만 추출
    metas.append({
        k: (
            str(v) if isinstance(v, (int, float, bool))
            else v or ""
            if isinstance(v, str)
            else json.dumps(v, ensure_ascii=False)
        )
        for k, v in item.items()
    })

    ids.append(f"doc_{i+1}")

# === DB 저장 ===
collection.add(documents=docs, metadatas=metas, ids=ids)
print(f" {len(docs)}개 문서가 DB에 추가되었습니다 → {DB_DIR}")
print("총 문서 수:", collection.count())