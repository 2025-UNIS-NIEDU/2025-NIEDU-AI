# NIEdu 📚
뉴스 기반 개인 맞춤형 문해력·시사 상식 학습 서비스

---

## 📌 프로젝트 소개
NIEdu는 뉴스 콘텐츠를 활용하여 사용자의 문해력과 시사 상식을 확장하는 교육 서비스입니다.
AI를 활용한 개인화 학습과 게이미피케이션 요소를 통해, 뉴스 소비를 학습 루틴으로 만들 수 있도록 돕습니다. 

뉴스 수집 → 벡터 임베딩 → RAG 기반 구조화 → KMeans 클러스터링 →  LLM 기반 코스명·키워드 생성 → 단계별 퀴즈 생성까지의 전 과정을 자동화하였습니다.  

---

## 🛠 Tech Stack  
- **Language** : Python 3.11  
- **Framework** : LangChain 
- **Database** : ChromaDB (Topic-based RAG)  
- **Clustering / Embedding** : KMeans · OpenAI *text-embedding-3-small*  
- **LLM** : OpenAI GPT-4o  
- **Infra** : Poetry  
- **External APIs** : DeepSearch News API · Google CSE API  
- **LLM API Server** : FastAPI 

---

## ⚙️ Core Features  

## 🧩 코스 생성 및 번들링
- 뉴스 본문을 **RAG 임베딩** 후 **주제별 클러스터링**
- 각 클러스터를 LLM이 분석해 **대표 코스명 생성 및 키워드(소주제) 분류 수행**
- 코스 단위 데이터는 퀴즈 파이프라인으로 전달되어 후속 학습에 사용

---

## 🧠 N 단계 (Background Stage)
- 뉴스에 등장하는 **핵심 개념·전문용어**를 추출 및 정의
- Google Custom Search API 기반으로 시의성 있는 **배경지식 카드** 자동 구성
- LLM이 개념 간 관계를 정리하여 **“용어–이슈–결과–영향”** 형태로 출력
→ 학습자의 **기본 이해도**에 맞춘 기초 정보 제공

---

## 💡 I 단계 (Intermediate Stage)
- **질문–회고(Question–Reflection)** 구조 구현
- LLM이 **뉴스 내용 근거의 정·오답 분석 및 피드백**을 자동 생성

---

## ✍️ E 단계 (Evaluation Stage)
- I단계 문제 중 일부를 변환하여 **비판적·확장적 사고형** 문제로 재구성
- **문장 완성형·추론형** 등 고급 문해력 평가용 문제로 변환
- LLM이 **의미 일치도·논리적 타당성** 기준으로 채점 및 피드백 생성

---

## 📂 Quiz Module Structure (`src/quiz`)

이 폴더는 **NIEdu 프로젝트의 퀴즈 자동 생성 및 평가 파이프라인**을 구성하는 핵심 모듈입니다.  
뉴스 요약 데이터를 기반으로 단계별(N/I/E) 학습 퀴즈를 자동 생성하고,  
사용자 답변 평가까지 포함하는 전체 로직이 파일별로 분리되어 있습니다.

---

```markdown
## 🧭 폴더 구조
```

src/
├── course/                         # 뉴스 코스(묶음) 생성 및 관리 모듈
│   ├── course_bundler.py           # 세션을 코스로 묶는 핵심 로직
│   ├── news_api.py                 # DeepSearch 뉴스 API 호출 및 파싱
│   ├── rag_builder.py              # 뉴스 RAG(Vector DB) 구축
│   ├── run_economy.py              # 경제 카테고리 뉴스 코스 생성
│   ├── run_politics.py             # 정치 카테고리 뉴스 코스 생성
│   ├── run_society.py              # 사회 카테고리 뉴스 코스 생성
│   ├── run_tech.py                 # 기술 카테고리 뉴스 코스 생성
│   ├── run_world.py                # 국제 카테고리 뉴스 코스 생성
│   └── **init**.py
│
└── quiz/                           # 퀴즈 및 단계별 학습 문제 생성 모듈
├── background_n.py             # N단계(기초) 배경지식 카드 생성
├── completion_e.py             # E단계(고급) 문장 완성형 문제 생성
├── completion_feedback_e.py    # 사용자 답변 피드백 생성
├── keyword_nie.py              # 핵심 키워드 추출 및 용어 카드 생성
├── multi_ni.py                 # N/I 단계 객관식 문제 생성
├── ox_n.py                     # N단계 OX 문제 생성
├── reflection_ie.py            # I/E 단계 회고형 질문 생성
├── select_session.py           # 세션 선택 및 메타데이터 로드
├── short_ie.py                 # I/E 단계 단답형 문제 생성
├── term_n.py                   # N단계 전문 용어 카드 생성
└── **init**.py

````

---

## 🚀 실행 방법
```bash
poetry install
poetry run python src/course/run_politics.py
````

---

📚 **Part of the NIEdu Project**
🔗 *AI-Driven News Learning Backend (RAG + LLM Integration)*

```
---
