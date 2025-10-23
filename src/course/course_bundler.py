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

# === 임베딩 함수 ===
embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api_key, model_name="text-embedding-3-small"
)

def generate_course_for_topic(topic: str):
    print(f"\n[{topic}] ChromaDB 불러오는 중...")

    DB_DIR = BASE_DIR / "data" / "db" / topic
    chroma_client = chromadb.PersistentClient(path=str(DB_DIR))

    # --- 컬렉션 감지 ---
    collections = chroma_client.list_collections()
    if not collections:
        print(f"[경고] {topic}: 컬렉션 없음. 스킵.")
        return

    collection_name = (
        collections[0] if isinstance(collections[0], str)
        else collections[0].get("name", f"{topic}_news")
    )

    collection = chroma_client.get_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )
    print(f"감지된 컬렉션: {collection_name}")

    # --- 데이터 불러오기 ---
    try:
        all_data = collection.get(include=["metadatas"], limit=5000)
        metadatas = all_data.get("metadatas", [])
    except Exception as e:
        print(f"[오류] {topic} 데이터 로드 실패: {e}")
        return

    if not metadatas:
        print(f"{topic} — 불러온 문서 없음 (docs 비어있음)")
        return

    docs = []
    for meta in metadatas:
        if not isinstance(meta, dict):
            continue
        filtered_meta = {k: v for k, v in meta.items() if k != "deepsearchId"}
        docs.append(sort_session_keys(filtered_meta))

    print(f"{topic} 뉴스 개수: {len(docs)}")

    # --- subTopic별 그룹화 ---
    grouped_by_sub = defaultdict(list)
    for d in docs:
        subTopic = d.get("subTopic", "기타") or "기타"
        grouped_by_sub[subTopic].append(d)

    # --- 9개 이상 문서가 포함된 subTopic만 필터링 ---
    filtered_subtopics = {k: v for k, v in grouped_by_sub.items() if len(v) >= 9}

    if not filtered_subtopics:
        print(f"[{topic}] 10개 이상 문서가 포함된 subTopic 없음 → 스킵")
        return
    
    output = []

    # --- 각 subTopic 단위로 코스 생성 ---
    for idx, (subTopic, group_news) in enumerate(filtered_subtopics.items(), start=1):
        for sid, news in enumerate(group_news, start=1):
            news["sessionId"] = sid
            group_news[sid - 1] = sort_session_keys(news)

        cleaned_sessions = []
        for s in group_news:
            filtered = {k: v for k, v in s.items() if k not in ("topic", "subTopic", "deepsearchId")}
            cleaned_sessions.append(sort_session_keys(filtered))    

        summaries = [
            f"- {n.get('headline', '')}: {n.get('summary', '')[:400]}"
            for n in group_news
        ]
        joined_summary = "\n".join(summaries)

        # --- 코스 설명 생성 ---
        prompt_course = f"""
출력은 반드시 JSON 형식으로 작성하세요.

당신은 뉴스 기반 학습 콘텐츠를 기획하는 에디터입니다.
아래 여러 뉴스 요약을 바탕으로 하나의 학습 코스로 묶으세요.

규칙:
1. courseName은 **최소 20자,25자 내외**로 **명사형으로 끝나는 구조**여야 하며, 문장처럼 끝나지 않습니다.  
   - 문어체이되, 딱딱한 학술 표현은 피하고 부드러운 흐름을 유지하세요.  
    + 문어체(글쓰기체)로 작성하되, 논문처럼 딱딱하지 않고 신문·칼럼처럼 자연스럽게 읽히게 하세요.
   - 필요하다면 은유적/상징적인 문구를 사용하세요.
2. courseName은 **동일 토픽 내의 다른 코스명과 단어가 중복되지 않게** 독창적으로 만드세요.
   - 이미 사용된 단어(예: '금융', '시장', '정책')이 포함된 제목은 피합니다.
3. courseDescription은 **90~110자 내외**로 **무엇을 중심으로 어떤 관점에서 학습하게 되는지**를 보여주세요.
   - 단순 요약이 아닌, 학습자가 얻을 인사이트 중심으로 서술하세요.
4. sub_tag는 **동일 토픽 내의 다른 코스들과 단어가 조금도 겹치지 않게 해주세요.**
이 뉴스 그룹의 세부 주제를 2~3개 단어로 요약한 해시태그 형식으로 작성합니다.  
   - 예: "#산업전환, #시장흐름, #정책방향"

출력 형식(JSON):
{{
  "courseName": string,
  "courseDescription": string,
  "sub_tag": string
}}

[뉴스 요약 목록]
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
            print(f"[{topic}] course 생성 실패: {e}")
            meta_course = {
                "courseName": f"{topic}_{subTopic}",
                "courseDescription": "자동 생성 실패 → 기본 설명",
                "sub_tag": ""
            }

        tag_candidates = []
        for k, v in meta_course.items():
            if "tag" in k.lower():
                if isinstance(v, list):
                    tag_candidates.extend(v)
                elif isinstance(v, str):
                    tag_candidates.append(v)

        clean_tags = sorted(set([t.strip() for t in tag_candidates if t.strip()]))
        subTopics = []
        for tag_str in clean_tags:
            for token in tag_str.replace(",", " ").split():
                clean_token = token.strip()
                if clean_token and clean_token not in subTopics:
                    subTopics.append(clean_token)

        course_data = OrderedDict([
            ("courseId", idx),
            ("topic", topic),
            ("subTopics", subTopics),
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

    print(f"{topic} → 코스 파일 저장 완료: {output_file.resolve()}")