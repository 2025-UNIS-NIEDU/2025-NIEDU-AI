# === 표준 라이브러리 ===
import os, json, time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# === 경로 설정 ===
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"
COURSE_DIR = BASE_DIR / "data" / "course_db"
FILTER_DIR = COURSE_DIR / "filtered"
FILTER_DIR.mkdir(parents=True, exist_ok=True)

# === 환경 변수 로드 ===
load_dotenv(ENV_PATH, override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === 임베딩 모델 (한번만 로드) ===
embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")

# === LLM 프롬프트 ===
PROMPT_FILTER = """
너는 뉴스 기반 학습 코스의 세션 검수자이다.
다음 세션(뉴스 제목) 중 정말로 주제와 전혀 관계없는 것만 골라라.

판단 기준:
- 사회, 정치, 경제, 외교, 기술, 정책 등 시사 흐름과 관련 있으면 무조건 True(유지)
- 단순 사건·사고, 연예, 날씨, 광고, 축제, 포토뉴스처럼 교육 가치가 전혀 없는 것만 False(제거)
- 인물 중심이라도 정책, 발언, 사회적 영향이 있으면 True로 둔다.

즉, “이건 아무리 봐도 학습용으로 쓸 수 없다” 싶은 것만 제거하라.

출력 형식(JSON Lines):
[
  {"courseName": "...", "is_educational": true or false, "reason": "..."},
  ...
]

주의 :
1. 출력 시 절대로 ```json 같은 마크다운 표시는 쓰지 말고,
2. 순수 JSON만 반환하라.
"""

# === 안전한 LLM 요청 함수 ===
def safe_request(prompt, retry=3, wait=3):
    for attempt in range(retry):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "너는 뉴스 코스의 품질 평가자다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[경고] 재시도 {attempt+1}/{retry}: {e}")
            time.sleep(wait)
    return None


# === 코스명 기반 필터링 ===
def filter_by_course_name(courses, batch_size=8):
    results = []
    for i in tqdm(range(0, len(courses), batch_size), desc="1단계: 코스명 LLM 필터링"):
        batch = courses[i:i+batch_size]
        names = "\n".join([f"- {c.get('courseName', '')}" for c in batch])
        prompt = f"{PROMPT_FILTER}\n\n{names}"

        resp = safe_request(prompt)
        if not resp:
            for c in batch:
                results.append({
                    "courseName": c.get("courseName", ""),
                    "is_educational": True,
                    "reason": "API 응답 실패 (임시 통과)"
                })
            continue

        try:
            parsed = json.loads(resp)
            if isinstance(parsed, list):
                results.extend(parsed)
            else:
                results.append(parsed)
        except Exception as e:
            print(f"[파싱 오류] {e}\n응답:\n{resp[:200]}...\n")
            for c in batch:
                results.append({
                    "courseName": c.get("courseName", ""),
                    "is_educational": True,
                    "reason": "JSON 파싱 실패 (임시 통과)"
                })
    return results

def compute_coherence(course, noise_threshold=0.4, return_removed=False):
    """
    한 코스 내 headline 임베딩 간 유사도를 계산하고,
    중심 벡터와의 cosine similarity가 낮은 세션을 제거한다.
    return_removed=True일 경우 제거된 headline 목록도 반환한다.
    """
    headlines = [s.get("headline", "") for s in course.get("sessions", []) if s.get("headline")]
    if len(headlines) < 2:
        return 0.0 if not return_removed else (0.0, [])

    # === headline 임베딩 ===
    embs = embedding_model.encode(headlines, convert_to_numpy=True, normalize_embeddings=True)

    # === 세션 간 평균 유사도 ===
    sims_matrix = cosine_similarity(embs)
    upper = np.triu_indices_from(sims_matrix, k=1)
    coherence_score = float(np.mean(sims_matrix[upper]))

    # === 중심 벡터(centroid) 계산 ===
    centroid = np.mean(embs, axis=0)
    centroid_sims = cosine_similarity(embs, centroid.reshape(1, -1)).flatten()

    # === 노이즈 세션 필터링 ===
    kept_sessions = []
    removed_sessions = []
    for i, s in enumerate(course["sessions"]):
        sim_score = float(centroid_sims[i])
        s["similarity_to_center"] = sim_score
        if sim_score >= noise_threshold:
            kept_sessions.append(s)
        else:
            removed_sessions.append({
                "headline": s.get("headline", ""),
                "similarity": sim_score
            })

    # === 코스 내 세션 업데이트 ===
    course["sessions"] = kept_sessions

    if return_removed:
        return coherence_score, removed_sessions
    else:
        return coherence_score
    
PROMPT_SESSION_REVIEW = """
너는 뉴스 기반 학습 코스의 검수자이다.
다음은 한 코스의 세션(뉴스 기사 제목) 목록이다.
코스명과 세션 제목들을 읽고, 주제 흐름에서 벗어나거나 관련성이 약한 세션만 골라라.

판단 기준:
1. 다음 유형만 False(제거)로 판단한다:
   - 연예, 사건·사고, 범죄, 재난, 단순 교통·화재 보도
   - 기업·제품·브랜드 홍보, 광고성 기사, 신제품 출시
   - 지역 축제·행사·공연·시상식 등 일회성 이벤트
   - 날씨, 계절성 풍경, 단순 통계·참석자 수·수상 인원 등 정보만 나열된 기사
   - 단순 언급이나 형식적 발언만 있고 정책·사회적 맥락이 없는 기사

출력 형식(JSON):
{
  "off_topic": ["headline1", "headline2", ...],
  "reason": "왜 자른건지 이유를 설명"
}
"""

def review_sessions_with_llm(course):
    """LLM이 코스 내 세션 headline을 검수하고 off-topic 세션을 제거"""
    headlines = "\n".join([f"- {s['headline']}" for s in course.get("sessions", [])])
    prompt = f"{PROMPT_SESSION_REVIEW}\n\n코스명: {course['courseName']}\n\n{headlines}"

    resp = safe_request(prompt)
    if not resp:
        return course  # 실패 시 그대로 반환

    # === 응답 전처리 (```json 등 제거) ===
    resp_clean = (
        resp.strip()
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:
        result = json.loads(resp_clean)
        off_topics = set(result.get("off_topic", []))
    except Exception as e:
        print(f"[세션 검수 JSON 파싱 오류] {e}\n{resp[:200]}...\n")
        return course

    # === 관련 없는 세션 제거 ===
    filtered_sessions = []
    removed_sessions = []
    for s in course["sessions"]:
        if s["headline"] in off_topics:
            removed_sessions.append(s["headline"])
        else:
            filtered_sessions.append(s)

    if removed_sessions:
        print(f"\n[{course['courseName']}] LLM 후처리로 제거된 세션 ({len(removed_sessions)}개):")
        for h in removed_sessions:
            print(f"  - {h}")

    course["sessions"] = filtered_sessions
    return course

# === 인덱스 재정렬 함수 ===
def reindex_courses(courses):
    """courseId 및 sessionId 재정렬"""
    for i, course in enumerate(courses):
        course["courseId"] = i + 1  # 1부터 시작
        for j, session in enumerate(course.get("sessions", [])):
            session["sessionId"] = j + 1
    return courses

# === 메인 파이프라인 ===
def main():
    for file in COURSE_DIR.glob("*.json"):
        print(f"\n{file.name} 처리 중...")

        # === 데이터 로드 ===
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # === 🔸정치 JSON 파일은 필터링 없이 그대로 복사 ===
        if "politics" in file.name.lower():
            out_path = FILTER_DIR / file.name
            with open(file, "r", encoding="utf-8") as src, open(out_path, "w", encoding="utf-8") as dst:
                dst.write(src.read())
            print(f"[정치 파일] 필터링 건너뜀 — 원본 복사 완료 → {out_path.name}")
            continue  # 다음 파일로 넘어감

        # === 데이터 로드 ===
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # === 정치 코스 제외 ===
        non_politics = [c for c in data if c.get("topic") != "정치"]

        # === 1️. LLM 코스 필터링 ===
        filter_results = filter_by_course_name(non_politics)
        result_map = {r["courseName"]: r for r in filter_results}

        print("\n=== LLM 필터링 결과 (요약) ===")
        print(json.dumps(filter_results, ensure_ascii=False, indent=2))

        # === 2️. 부적절 코스 제거 ===
        filtered, removed = [], []
        for c in data:
            course_name = c["courseName"]
            result = result_map.get(course_name, {})
            if result.get("is_educational", True):
                filtered.append(c)
            else:
                removed.append({
                    "courseName": course_name,
                    "reason": result.get("reason", "이유 없음")
                })

        print(f"\n학습용 {len(filtered)}개 / 제거 {len(removed)}개")

        if removed:
            print("\n=== 제거된 코스 목록 ===")
            for r in removed:
                print(f"  - {r['courseName']}  ({r['reason']})")
            print("-" * 50)

        # === 3️. 코스 일관성 + 세션 노이즈 제거 ===
        coherence_scores = []
        print("\n=== 코스별 세션 노이즈 제거 및 일관성 점수 계산 ===")

        for c in tqdm(filtered, desc="2단계: 세션 정제 및 점수 계산"):
            score, removed_sessions = compute_coherence(c, noise_threshold=0.55, return_removed=True)
            coherence_scores.append({"courseName": c["courseName"], "score": score})

            if removed_sessions:
                print(f"\n[{c['courseName']}] 제거된 세션 ({len(removed_sessions)}개):")
                for r in removed_sessions:
                    print(f"  - {r['headline']} ({r['similarity']:.3f})")

        coherence_scores.sort(key=lambda x: x["score"])

        # === 4. LLM 세션 후처리 ===  ← 추가
        print("\n=== 3단계: LLM 세션 일관성 재검수 ===")
        refined = []
        for c in tqdm(filtered, desc="3단계: LLM 세션 검수"):
            reviewed = review_sessions_with_llm(c)
            if reviewed.get("sessions"):  # 세션이 남아있는 경우만 유지
                refined.append(reviewed)
            else:
                print(f"{c['courseName']} 모든 세션이 제거되어 제외됨")
        filtered = refined

        # === 5. 세션 개수가 너무 적은 코스 제거 === ★ 여기에 추가 ★
        final_courses = []
        removed_short = []
        for c in filtered:
            if len(c.get("sessions", [])) < 4:
                removed_short.append({
                    "courseName": c["courseName"],
                    "reason": f"세션 {len(c.get('sessions', []))}개로, 학습용 기준(5개) 미달"
                })
            else:
                final_courses.append(c)
        filtered = final_courses

        if removed_short:
            print(f"\n=== 세션 부족으로 제거된 코스 ({len(removed_short)}개) ===")
            for r in removed_short:
                print(f"  - {r['courseName']} ({r['reason']})")
            print("-" * 50)

        # === 6. 코스별 점수 출력 ===
        print("\n=== 코스별 세션 일관성 점수 ===")
        for s in coherence_scores:
            print(f"  - {s['courseName']}: {s['score']:.3f}")
        print("===============================")

        # === 7. 가장 일관성 낮은 코스 표시 ===
        worst = coherence_scores[0] if coherence_scores else None
        if worst:
            print(f"\n일관성 가장 낮은 코스: {worst['courseName']} ({worst['score']:.3f})")

        # === 8. 인덱스 재정렬 ===
        reindexed_filtered = reindex_courses(filtered)

        # 불필요한 similarity 필드 제거는 reindexed_filtered 대상으로 수행
        for c in reindexed_filtered:
            for s in c.get("sessions", []):
                s.pop("similarity_to_center", None)

        # 정치 코스(원본) + 정제된 비정치 코스 병합
        merged = reindexed_filtered

        # === 9. 결과 저장 ===
        out_path = FILTER_DIR / f"{file.name}"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"\n필터링 및 노이즈 제거 완료 → {out_path.name}")

    print("\n모든 파일 처리 완료.")

if __name__ == "__main__":
    print("=== 뉴스 코스 필터링 시작 ===")
    main()
    print("=== 모든 코스 처리 완료 ===")