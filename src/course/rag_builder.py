import os
import json
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from collections import OrderedDict
from datetime import datetime

# 0️. 기본 설정
today = datetime.now().strftime("%Y-%m-%d")

BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"
BACKUP_DIR = BASE_DIR / "data" / "backup"
DB_ROOT = BASE_DIR / "data" / "rag_db"

BACKUP_DIR.mkdir(parents=True, exist_ok=True)
DB_ROOT.mkdir(parents=True, exist_ok=True)

# 1️. 환경 변수 및 임베딩 설정
load_dotenv(dotenv_path=ENV_PATH, override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="jhgan/ko-sroberta-multitask"
)

# 2️. 키 정렬 함수 (deepsearchId, topic, subTopic 우선)
def sort_deepsearchId_keys(session: dict) -> dict:
    """deepsearchId → topic → subTopic 순으로 고정, 나머지는 알파벳 순 정렬"""
    if not isinstance(session, dict):
        return session

    ordered = []
    for key in ("deepsearchId", "topic", "subTopic"):
        if key in session:
            ordered.append((key, session[key]))

    remaining = sorted(
        [(k, v) for k, v in session.items() if k not in ("deepsearchId", "topic", "subTopic")],
        key=lambda x: x[0].lower()
    )

    return dict(ordered + remaining)

# 3️. 백업 폴더 내 JSON 파일 검색
json_files = list(BACKUP_DIR.glob(f"*{today}*.json"))
print(f"{len(json_files)}개의 JSON 파일을 감지했습니다.")

# 4️. 파일별 변환 및 DB 저장
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

    # 5️. 기사별 문서 변환
    for i, item in enumerate(articles):
        try:
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

            # 임베딩용 텍스트 구성 (자연어 기반)
            page_content = f"""
            [기사 제목] {clean_dict.get('headline', '')}

            [주제] {clean_dict.get('topic', '')}
            [세부 주제] {clean_dict.get('subTopic', '')}

            [요약] {clean_dict.get('summary', '')}

            [언론사] {clean_dict.get('publisher', '')}
            [게시일] {clean_dict.get('publishedAt', '')}

            [기사 URL] {clean_dict.get('contentUrl', '')}
            [썸네일] {clean_dict.get('thumbnailUrl', '')}

            [DeepSearch ID] {clean_dict.get('deepsearchId', '')}
            """

            docs.append(page_content.strip())  # 실제 임베딩에 사용
            metas.append(clean_dict)           # 전체 메타데이터 저장
            ids.append(f"{topic_name}_{i+1}")

        except Exception as e:
            print(f"[경고] {topic_name} 문서 {i} 처리 중 오류: {e}")

    # 6️. DB에 저장
    if docs:
        try:
            collection.add(documents=docs, metadatas=metas, ids=ids)
            print(f"{len(docs)}개 문서를 DB에 추가 완료 → {topic_db_path}")
            print(f"현재 컬렉션 문서 수: {collection.count()}")

            # 추가된 데이터 샘플 1~2개만 확인
            sample = collection.get(limit=2, include=["documents", "metadatas"])
            print(f"샘플 문서 2개 미리보기 ({topic_name}):")
            for idx, doc in enumerate(sample["documents"]):
                meta = sample["metadatas"][idx]
                print(f"  ├─ [{meta.get('subTopic', 'N/A')}] {meta.get('headline', '')[:50]}...")
            print("-" * 80)

        except Exception as e:
            print(f"[오류] {topic_name} DB 저장 중 문제 발생: {e}")
    else:
        print(f"[주의] {topic_name}에 유효한 문서가 없습니다.")

# 7️. subTopic 검증
print("\n=== subTopic 필드 검증 ===")
for topic_dir in DB_ROOT.iterdir():
    if topic_dir.is_dir():
        client = chromadb.PersistentClient(path=str(topic_dir))
        collection = client.get_or_create_collection(
            name=f"{topic_dir.name}_news",
            embedding_function=embedding_fn
        )
        data = collection.get(include=["metadatas"])
        if not data["metadatas"]:
            print(f"{topic_dir.name}: 메타데이터 없음")
            continue

        subs = [m.get("subTopic", None) for m in data["metadatas"] if m]
        unique_subs = set(s for s in subs if s)
        print(f"{topic_dir.name}: subTopic 필드 확인 ({len(unique_subs)}개 고유값)")
        print(f"예시: {list(unique_subs)[:5]}")

print("\n모든 토픽 DB 저장 및 검증이 완료되었습니다.")

# 7️. RAG DB 문서 수 확인
print("\n=== 각 토픽별 문서 수 확인 ===")
BASE_DIR = Path(__file__).resolve().parents[2]
DB_ROOT = BASE_DIR / "data" / "db"

for topic_dir in DB_ROOT.iterdir():
    if topic_dir.is_dir():
        client = chromadb.PersistentClient(path=str(topic_dir))
        collection = client.get_or_create_collection(
            name=f"{topic_dir.name}_news"
        )
        count = collection.count()
        print(f"[{topic_dir.name}] 문서 수: {count}")