# 1. Python 3.11 기반 이미지 사용
FROM python:3.11-slim

# 2. 컨테이너 내 작업 디렉토리 설정
WORKDIR /app

# 3. Poetry 설치 및 가상환경 비활성화
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false

# 4. 프로젝트 설정 파일 복사 (의존성 레이어 캐싱)
COPY pyproject.toml poetry.lock* ./

# 5. 프로덕션용 패키지만 설치 (dev 그룹 제외)
RUN poetry install --without dev --no-root

# 6. 소스 코드 복사
COPY . .

# 7. FastAPI 실행 (포트 8000)
EXPOSE 8000
# main.py 파일의 실제 경로에 맞게
CMD ["poetry", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]