import os, json, logging
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from collections import OrderedDict
from datetime import datetime

log = logging.getLogger(__name__) 

def build_rag_data():
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

    # 3️. 백업 폴더 내 JSON 파일 중 최신 날짜만 유지
    json_files_all = sorted(BACKUP_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)
    if not json_files_all:
        log.warning("백업 폴더에 JSON 파일이 없습니다.")
        return

    latest_date = json_files_all[0].stem.split("_")[-1].split(".")[0]
    json_files = [f for f in json_files_all if latest_date in f.name]
    log.info(f"📅 최신 날짜({latest_date}) 기준 {len(json_files)}개 JSON 처리")

    # 기존 DB 초기화 (컬렉션 단위 삭제)
    for topic_dir in DB_ROOT.iterdir():
        if topic_dir.is_dir():
            client = chromadb.PersistentClient(path=str(topic_dir))
            collection_name = f"{topic_dir.name}_news"
            try:
                client.delete_collection(name=collection_name)
                log.info(f"[{topic_dir.name}] 기존 컬렉션 초기화 완료")
            except Exception as e:
                log.warning(f"[{topic_dir.name}] 컬렉션 초기화 실패: {e}")

    # 4️. 최신 JSON 파일별 변환 및 DB 저장
    for json_file in json_files:
        topic_name = json_file.stem.split("_")[0]
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
            log.error(f"[{topic_name}] JSON 로드 실패: {e}")
            continue

        docs, metas, ids = [], [], []

        # 5️. 기사별 문서 변환
        for i, item in enumerate(articles):
            try:
                ordered_item = sort_deepsearchId_keys(item)

                clean_dict = OrderedDict()
                for k, v in ordered_item.items():
                    if v is None:
                        v = ""
                    elif isinstance(v, (list, dict)):
                        v = json.dumps(v, ensure_ascii=False)
                    else:
                        v = str(v)
                    clean_dict[k] = v

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

                docs.append(page_content.strip())
                metas.append(clean_dict)
                ids.append(f"{topic_name}_{i+1}")

            except Exception as e:
                log.warning(f"[{topic_name}] 문서 {i} 변환 중 오류: {e}")
                continue

        # 6️. DB에 저장
        if docs:
            try:
                collection.add(documents=docs, metadatas=metas, ids=ids)
                log.info(f"[{topic_name}] {len(docs)}개 문서 추가 완료")
            except Exception as e:
                log.error(f"[{topic_name}] DB 저장 중 문제 발생: {e}")

    # 7️. subTopic 검증
    for topic_dir in DB_ROOT.iterdir():
        if topic_dir.is_dir():
            client = chromadb.PersistentClient(path=str(topic_dir))
            collection = client.get_or_create_collection(
                name=f"{topic_dir.name}_news",
                embedding_function=embedding_fn
            )
            data = collection.get(include=["metadatas"])
            if not data["metadatas"]:
                continue
            subs = [m.get("subTopic", None) for m in data["metadatas"] if m]
            unique_subs = set(s for s in subs if s)
            log.info(f"[{topic_dir.name}] subTopic {len(unique_subs)}개 확인")

    # 8. RAG DB 문서 수 확인
    for topic_dir in DB_ROOT.iterdir():
        if topic_dir.is_dir():
            client = chromadb.PersistentClient(path=str(topic_dir))
            collection = client.get_or_create_collection(
                name=f"{topic_dir.name}_news"
            )
            count = collection.count()
            log.info(f"[{topic_dir.name}] 문서 수: {count}")

#  실행
if __name__ == "__main__":
    build_rag_data()