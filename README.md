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
- **Framework** : LangChain · LangGraph  
- **Database** : ChromaDB (Topic-based RAG)  
- **Clustering / Embedding** : KMeans · OpenAI *text-embedding-3-small*  
- **LLM** : OpenAI GPT-4o  
- **Infra** : Poetry  
- **External APIs** : DeepSearch News API · Google CSE API  
- **LLM API Server** : FastAPI 

---

## ⚙️ Core Features  

## 🧩 코스 생성 및 번들링
- 뉴스 본문을 **RAG 임베딩** 후 KMeans 알고리즘으로 **주제별 클러스터링**
- 각 클러스터를 LLM이 분석해 **대표 코스명 생성 및 키워드(소주제) 분류 수행**
- 코스 단위 데이터는 퀴즈 파이프라인으로 전달되어 후속 학습에 사용

---

## 🧠 N 단계 (Background Stage)
- Google Custom Search API 로 시의성 높은 데이터를 검색해 **배경지식 자동 구성**
- LLM이 관련 개념·이슈를 구조적으로 정리하여 **이슈명·원인·결과·영향** 형태로 출력

---

## 💡 I 단계 (Intermediate Stage)
- LangGraph 기반 **질문–회고(Question–Reflection)** 구조 구현
- LLM이 **뉴스 내용 근거의 정·오답 분석 및 피드백**을 자동 생성

---

## ✍️ E 단계 (Evaluation Stage)
- 문장 완성·추론형 문제를 통한 **비정형 문해력 평가**
- LLM이 **의미 일치도·문법 정확성 기준으로 채점 및 피드백 제공**

---

## 📂 Repository Structure 
```bash
src/
┣ course/ # 뉴스 수집 → 임베딩 → KMeans 클러스터링 → 코스 생성
┣ quiz/   # LangGraph 기반 퀴즈 파이프라인 (N · I · E 단계)
┗ api/    # 백엔드 요청 처리용 FastAPI 서버
```

---

## 🚀 Run  
```bash
poetry install
```

📚 Part of the NIEdu project
🔗 AI-Driven News Learning Backend (RAG + LLM Integration)