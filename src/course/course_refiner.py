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

# === 코스 검수 ===
PROMPT_FILTER = """
너는 뉴스 기반 학습 코스의 기획 검수자이다.
다음은 여러 개의 '코스 이름' 목록이다.
각 코스가 학습 주제로서 타당한지 판단하라.

---

### 판단 기준

#### **학습적 가치가 있는 코스(True)**
다음 중 하나 이상에 해당하면 True로 판단한다.

1. **사회 구조·정책 맥락** — 제도나 정책의 방향성, 사회 구조적 변화와 관련된 주제  
2. **인과관계와 변화 흐름** — 기술·경제 변화가 사회 전반에 미치는 영향  
3. **세대·가치관·사회 인식** — 세대 간 인식 차이, 사회문화적 변화를 다루는 주제  
4. **논쟁과 균형 시각** — 찬반 구조나 사회적 갈등을 균형 있게 탐구하는 주제  
5. **국제 관계·경제 구조** — 세계 경제, 외교 관계, 국제 협력 구조를 다루는 주제  
6. **사회 문제·제도 변화** — 복지, 교육, 노동, 주거 등 사회 시스템 변화와 연관된 주제  
7. **기술과 사회의 접점** — 기술 발전이 인간 삶, 교육, 사회 구조에 미치는 의미  
8. **공공성·윤리·책임** — 윤리, 정책, ESG, 사회적 책임 등 공공 가치 논의  
9. **맥락적 서사(변화의 배경)** — 사회 변화의 원인과 결과를 연결하는 서사 구조  
10. **감정·가십 중심 여부(부정)** — 자극적 감정이나 개인 비난보다 사회적 성찰을 유도하는 주제

---

#### **학습적 가치가 낮은 코스(False)**
다음 중 하나라도 두드러지면 False로 판단한다.

- 인물 중심의 단순 발언·소감·논란 중심  
- 사건·사고, 범죄, 재난, 화재, 교통, 날씨 등 단발성 보도  
- 기업·브랜드 홍보, 신제품 출시, 행사·축제·시상식 등 상업성 기사  
- 감정적 대립, 비난, 선정적 제목 중심으로 구조·정책 맥락이 약한 경우  
- 단순 정치 공방, 여야 비난 구도만 반복되는 경우  

---


출력 형식(JSON Lines):
[
  {"courseName": "코스명", "is_educational": true or false, "reason": "판단 근거"}
]

주의:
1. 절대로 ```json 같은 마크다운 표시는 쓰지 말고,
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
**코스의 주제 흐름을 유지하면서**, 해당 주제와 관련성이 낮거나 완전히 벗어난 세션만 골라라.

---

판단 기준:

1️. **유지(True)**  
    - 코스의 주제(코스명)와 직접적 혹은 구조적으로 연결된 기사  
    - 정책, 제도, 경제·사회 구조, 외교·산업 변화 등 시사적 맥락이 있는 기사  
    - 인물 중심이라도 코스 주제와 연관된 발언, 정책 논의, 사회적 의미가 있으면 유지  

2️. **제거(False)**  
    - 다음 항목에 하나라도 해당되면 반드시 제거한다.
    [코스 주제와 무관]
    - 코스 주제와 직접 관련이 없음 
    - 동일한 주제·내용을 반복하는 경우 대표 1개만 유지
    [학습용 부적합]
    - 기업·제품·서비스·브랜드 홍보성 기사
    - 신제품 출시, 프로모션, 마케팅 캠페인, 인터뷰 중심 기사 포함
    - 일회성 이벤트/행사 중심 기사
    - 지역 축제, 시상식, 콘서트, 전시회, 단체 행사 등
    - 단순 정보·통계·참석자 수 중심 기사
    - 수상 인원, 참여 인원, 순위 발표 등 맥락 없는 숫자 나열형 기사
    - 정책·사회적 의미 없이 인물·발언만 전달하는 기사
    - 연예·스포츠·날씨·재난 등 비학습성 카테고리 기사
    - 사회 구조나 정책적 시사점이 없는 경우 모두 제거
    - 유사 내용의 반복 기사 중 중복성 높음
---

참고 예시 (학습용 부적합):

- 정치: ○○의원, 지역구 복지시설 방문…주민들과 간담회  
  → 표면상 복지정책 관련처럼 보이지만, 개인 활동 중심 기사로 제도·정책 변화 맥락이 없음  

- 경제: 대기업 ○○, 협력사 간담회 열고 상생협력 강조  
  → ‘상생’이라는 단어가 정책처럼 보이나, 기업 홍보성 기사로 구조적 경제 논의 없음  

- 사회: 서울시, 벚꽃축제 개막…시민 50만 운집  
  → 행정이 언급되지만 정책 변화 없이 단순 행사 중심 기사  

- 국제: 한일 정상, 짧은 환담…우호 분위기 조성  
  → 외교 주제 같지만, 협정·합의 등 실질 내용 없이 분위기 묘사 중심이면 학습 가치 낮음  

- 기술: ○○대 연구팀, 신기술 개발 성공…상용화 기대  
  → 기술 자체는 중요하나 사회·산업 구조적 의미가 부재한 단순 성과 기사

---

출력 형식(JSON):

{
  "off_topic": ["headline1", "headline2", ...],
  "reason": "이 기사들이 코스 주제와 관련이 약하거나 맥락이 다르다고 판단한 이유를 간단히 설명"
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

        # === 5. 세션 개수가 너무 적은 코스 제거 === 
        final_courses = []
        removed_short = []
        for c in filtered:
            if len(c.get("sessions", [])) < 3:
                removed_short.append({
                    "courseName": c["courseName"],
                    "reason": f"세션 {len(c.get('sessions', []))}개로, 학습용 기준(3개) 미달"
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