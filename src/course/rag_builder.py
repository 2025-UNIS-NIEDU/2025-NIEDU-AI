import os
import json
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from collections import OrderedDict

# === 경로 및 환경설정 ===
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"
BACKUP_DIR = BASE_DIR / "data" / "backup"
DB_ROOT = BASE_DIR / "data" / "db"

BACKUP_DIR.mkdir(parents=True, exist_ok=True)
DB_ROOT.mkdir(parents=True, exist_ok=True)

# === 환경 변수 로드 ===
load_dotenv(dotenv_path=ENV_PATH, override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === 임베딩 설정 ===
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)

# === deepsearchId, topic 고정 + 나머지는 알파벳 순 ===
def sort_deepsearchId_keys(session: dict) -> dict:
    """deepsearchId → topic 고정, 나머지는 알파벳 순으로 정렬"""
    if not isinstance(session, dict):
        return session

    ordered = []
    # 1️. deepsearchId, topic 먼저 배치
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

# === 백업 폴더 내 JSON 파일 탐색 ===
json_files = list(BACKUP_DIR.glob("*.json"))
print(f"{len(json_files)}개의 JSON 파일을 감지했습니다.")

# === 파일별 처리 ===
for json_file in json_files:
    topic_name = json_file.stem.split("_")[0]
    print(f"\n[{topic_name}] 변환 중...")

    topic_db_path = DB_ROOT / topic_name
    topic_db_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(topic_db_path))
    collection = client.get_or_create_collection(
        name=f"{topic_name}_news",
        embedding_function=embedding_fn
    )

    # === JSON 로드 ===
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        articles = raw.get("articles", raw.get("data", []))
    except Exception as e:
        print(f"[오류] {topic_name} JSON 로드 실패: {e}")
        continue

    docs, metas, ids = [], [], []

    for i, item in enumerate(articles):
        try:
            # 정렬 적용
            ordered_item = sort_deepsearchId_keys(item)

            # NoneType, list, dict 방지
            clean_dict = OrderedDict()
            for k, v in ordered_item.items():
                if v is None:
                    v = ""
                elif isinstance(v, (list, dict)):
                    v = json.dumps(v, ensure_ascii=False)
                else:
                    v = str(v)
                clean_dict[k] = v

            docs.append(json.dumps(clean_dict, ensure_ascii=False))
            metas.append(clean_dict)
            ids.append(f"{topic_name}_{i+1}")

        except Exception as e:
            print(f"[경고] {topic_name} 문서 {i} 처리 중 오류: {e}")

    # === DB 저장 ===
    if docs:
        try:
            collection.add(documents=docs, metadatas=metas, ids=ids)
            print(f" {len(docs)}개 문서를 DB에 추가 완료 → {topic_db_path}")
            print(f" 현재 컬렉션 문서 수: {collection.count()}")
        except Exception as e:
            print(f"[오류] {topic_name} DB 저장 중 문제 발생: {e}")
    else:
        print(f"[주의] {topic_name}에 유효한 문서가 없습니다.")

print("\n🎉 모든 토픽 DB 저장이 완료되었습니다.")