# NIEdu 📚
뉴스 기반 개인 맞춤형 문해력·시사 상식 학습 서비스

---

## 📌 프로젝트 소개
NIEdu는 뉴스 콘텐츠를 활용하여 사용자의 문해력과 시사 상식을 확장하는 교육 서비스입니다.
AI를 활용한 개인화 학습과 게이미피케이션 요소를 통해, 뉴스 소비를 학습 루틴으로 만들 수 있도록 돕습니다. 

뉴스 수집 → 벡터 임베딩 → RAG 기반 구조화 → KMeans 클러스터링 →  LLM 기반 코스명·키워드 생성 → 단계별 퀴즈 생성까지의 전 과정을 자동화하였습니다.  

---

## 🛠 Tech Stack  
Language: Python 3.11  
Framework: FastAPI, LangChain, LangGraph  
Database: ChromaDB (Topic-based RAG)  
Clustering / Embedding: KMeans, OpenAI text-embedding-3-small  
LLM: OpenAI GPT-4o  
Infra / Tools: Poetry, Docker  
External APIs: DeepSearch News API, Google CSE API  

---

## ⚙️ Core Features  

#### 🧩 코스 생성 및 번들링
- 뉴스 본문을 **RAG 임베딩** 후 KMeans 알고리즘으로 **주제별 클러스터링**  
- 각 클러스터는 LLM을 통해 **대표 코스명과 키워드(소주제)**로 자동 요약  
- 생성된 코스는 학습 단위로 구조화되어 퀴즈 파이프라인에 전달  

---

#### 🧠 N 단계 (Background Stage)
- Google CSE API를 이용해 관련 뉴스를 검색하고 **배경지식 자동 생성**  
- **시의성**이 높은 뉴스 속 개념과 이슈를 LLM 이 구조적으로 해석하여 사회적 맥락과 흐름을 직관적으로 이해할 수 있도록 지원

---

#### 💡 I 단계 (Intermediate Stage)
- LangGraph 기반 **질문–회고 루프(Question–Reflection Loop)** 구성  
- LLM이 **뉴스 내용에 근거한 정답·오답 피드백**을 제공하여  
  학습자의 이해 수준을 분석하고 자기 점검을 유도  

---

#### ✍️ E 단계 (Evaluation Stage)
- 문장 완성·추론형 문제를 통해 **비정형 문해력 평가** 수행  
- LLM이 **의미 일치도 및 문법 정확성 기준으로 채점**  
  → 뉴스 맥락에 기반한 **즉각적 피드백** 제공  
- 학습자의 논리적 사고력과 표현력을 종합적으로 진단  

---

## 📂 Repository Structure 
```bash
src/
┣ course/ # 뉴스 수집 → 임베딩 → KMeans 클러스터링 → 코스 생성
┣ quiz/ # LangGraph 기반 퀴즈 파이프라인 (N · I · E 단계)
┗ api/ # FastAPI 엔드포인트 / 응답 포맷 정의
```

---

## 🚀 Run  
```bash
poetry install
```

📚 Part of the NIEdu project
🔗 AI-Driven News Learning Backend (RAG + LLM Integration)