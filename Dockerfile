# 1. Python 3.12 기반 이미지 사용 (Poetry와 호환)
FROM python:3.12-slim

# 2. 컨테이너 내 작업 디렉토리 설정
WORKDIR /app

# 3. Poetry 및 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y curl && \
    pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false

# 4. 프로젝트 설정 파일 복사
COPY pyproject.toml poetry.lock* ./

# 5. 프로덕션용 패키지만 설치 (dev 제외)
RUN poetry install --without dev --no-root --no-interaction --no-ansi

# 6. 소스 코드 복사
COPY . .

# 7. FastAPI 실행 (포트 8000)
EXPOSE 8000
CMD ["poetry", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
