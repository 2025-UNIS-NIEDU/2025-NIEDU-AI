2025-NIEdu-AI/
│
├── src/                                     # 실제 서비스 코드 전부 들어가는 폴더 (import 기준 루트)
│   │
│   ├── course/                              # 코스 생성 / 뉴스 처리 / RAG / 카드 생성 등 데이터 파이프라인
│   │   ├── course_pipeline.py               # 코스 전체 흐름 관리 (뉴스 → RAG → 번들 → 카드)
│   │   ├── news_api.py                      # 외부 뉴스 API 호출 (DeepSearch, NewsAPI 등)
│   │   ├── rag_builder.py                   # 뉴스 본문 벡터화 + RAG 생성 (Embedding & Retrieval)
│   │   ├── course_bundler.py                # 여러 뉴스 묶어서 하나의 "코스" 단위로 번들링
│   │   ├── card_generator.py                # 번들링된 데이터를 카드 / 카드리스트 형태로 가공
│   │   └── __init__.py                      # 패키지 인식용 (비워둬도 됨)
│   │
│   ├── quiz/                                # LangGraph 기반 퀴즈 시스템
│   │   ├── graph_skeleton.py                # LangGraph 기본 구조(스켈레톤) 정의
│   │   ├── n_stage_nodes.py                 # N단계(기초) 노드 정의 (배경지식 + 퀴즈 SubNode)
│   │   ├── i_stage_nodes.py                 # I단계(중간) 노드 정의 (퀴즈 + 회기)
│   │   ├── e_stage_nodes.py                 # E단계(고급) 노드 정의 (단답형 + 문장완성 flow)
│   │   ├── quiz_pipeline.py                 # 각 단계 그래프 실행 제어 (build + run)
│   │   └── __init__.py
│   │
│   ├── api/                                 # FastAPI 서버 계층 (사용자와 직접 통신)
│   │   ├── main.py                          # 서버 entry point — FastAPI 실행 및 엔드포인트 정의
│   │   ├── schemas.py                       # 요청(Request) / 응답(Response) 모델 (Pydantic)
│   │   ├── response_formatter.py            # 내부 dict 결과를 ApiResponse 형태로 통합 포맷팅
│   │   └── __init__.py
│   │
│   ├── configs/                             # 환경 설정용 — 규모 커지면 유지, 작을 땐 생략 가능
│   │   └── settings.py                      # API 키, 모델명, 경로, 환경 변수 등 설정값 관리
│   │
│   ├── run_pipeline.py                      # 로컬 테스트용: 전체 코스+퀴즈 파이프라인 한 번 실행해보기
│   └── __init__.py
│
├── tests/                                   # 단위 테스트용 — 개발 중이면 나중에 추가해도 됨
│   └── test_api.py                          # FastAPI 엔드포인트 테스트 등
│
├── architecture.md                          # 폴더 구조 설명│
├── Dockerfile                               # 배포용 도커 설정 (환경 통일 + 서버 실행 자동화)
├── poetry.lock                              # 패키지 버전을 확정해서 재현 가능한 환경을 만드는 잠금(lock) 파일                         
└── pyproject.toml (또는 setup.py)            # Poetry / 패키징 설정 파일 (의존성, entry point 등)
