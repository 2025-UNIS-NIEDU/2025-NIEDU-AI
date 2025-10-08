# NIEdu 📚
**뉴스 기반 개인 맞춤형 문해력·시사 상식 학습 서비스**

</div>

---

### 📌 프로젝트 소개  
**NIEdu**는 뉴스 콘텐츠를 활용해 사용자의 문해력과 시사 상식을 확장하는 교육 서비스입니다.  
본 레포지토리는 **시사 학습을 위한 End-to-End AI 학습 파이프라인**을 담당합니다.  
뉴스 수집부터 RAG 기반 정보 구조화, LLM을 활용한 코스 생성·번들화,  
그리고 단계별(N · I · E) 퀴즈 자동 생성을 통해 개인 맞춤형 학습 루틴을 제공합니다.  

단계별 구조는 학습자의 문해력 수준에 맞춰 난이도를 점진적으로 조정하도록 설계되었으며,  
뉴스 콘텐츠의 복잡도를 완화해 누구나 시사 이슈에 쉽게 접근할 수 있도록 돕는 것을 목표로 합니다.

---

### 🛠 Tech Stack  
Language: Python 3.11  
Framework: FastAPI, LangChain  
Database: ChromaDB (Topic-based RAG)  
LLM / Embedding: OpenAI GPT-4o-mini, text-embedding-3-small  
Infra / Tools: Poetry, Docker (예정)  
External: DeepSearch API  

---

### 📂 Repository Structure  
```

src/
┣ course/   # 뉴스 수집 → 벡터화 → 코스 생성
┣ quiz/     # LangGraph 기반 퀴즈 생성
┗ api/      # FastAPI 엔드포인트 / 응답 포맷

````

---

### 🚀 실행 방법  
```bash
poetry install
poetry run python src/run_pipeline.py
````

---

📚 *Part of the NIEdu project*
🔗 *AI-Driven News Learning Backend (LLM Integration)*

</div>
```