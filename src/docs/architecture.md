2025-NIEdu-AI/
│
├── src/
│   │
│   ├── course/                              # 뉴스 → 코스 → 카드로 이어지는 데이터 생성 파이프라인
│   │   ├── course_pipeline.py               # 전체 흐름 제어 (뉴스 → RAG → 번들링 → 카드)
│   │   ├── news_api.py                      # 외부 뉴스 API 호출 (DeepSearch, NewsAPI 등)
│   │   ├── rag_builder.py                   # 뉴스 본문 벡터화 + RAG 구축 (임베딩 & 검색)
│   │   ├── course_bundler.py                # 여러 뉴스를 묶어 코스로 구성
│   │   ├── card_generator.py                # 코스를 카드 형태로 가공 (썸네일, 설명 등)
│   │   └── __init__.py
│   │
│   ├── quiz/                                # LangGraph 기반 퀴즈 생성 및 피드백 모듈
│   │   ├── graph_skeleton.py                # LangGraph 기본 구조 (노드 및 엣지 스켈레톤)
│   │   ├── n_stage_nodes.py                 # N단계(기초) 노드 (배경지식 및 초급 퀴즈)
│   │   ├── i_stage_nodes.py                 # I단계(중급) 노드 (회기형/추론형 퀴즈)
│   │   ├── e_stage_nodes.py                 # E단계(고급) 노드 (문장완성/서술형 평가)
│   │   ├── quiz_pipeline.py                 # 각 단계 그래프 실행 제어 및 통합 실행
│   │   └── __init__.py
│   │
│   ├── api/                                 # 🔹 LLM → 백엔드 API 연동 계층
│   │   ├── client.py                        # 백엔드 API 요청 (POST/GET) 전송 유틸
│   │   ├── endpoints.py                     # 엔드포인트 URL 및 API 경로 관리
│   │   ├── utils.py                         # 요청 헤더, 토큰 관리, 에러 처리
│   │   └── __init__.py
│   │
│   ├── configs/
│   │   └── settings.py                      # API 키, 모델명, 경로 등 환경 설정값 관리
│   │
│   ├── run_pipeline.py                      # 로컬 실행용: 전체 코스 + 퀴즈 파이프라인 실행 스크립트
│   └── __init__.py
│
├── tests/
│   └── test_quiz_pipeline.py                # 퀴즈 노드 및 파이프라인 로직 테스트
│
├── docs/
│   └── architecture.md                      # 구조/흐름도/명세서 정리 문서
│
├── poetry.lock
└── pyproject.toml                           # Poetry 의존성 및 패키징 설정