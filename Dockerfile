# 1. Python 3.11 기반 이미지 사용
FROM python:3.11-slim

# 2. 컨테이너 내 작업 디렉토리 설정
WORKDIR /app

# 3. PyTorch 설치 (CPU 버전) - 용량이 크므로 먼저 설치하여 레이어 캐시 활용
RUN pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# 4. requirements.deploy.txt 파일 복사
COPY requirements.deploy.txt .

# 5. 나머지 프로덕션용 패키지 설치
RUN pip install --no-cache-dir -r requirements.deploy.txt

# 6. 소스 코드 복사
COPY . .

# 7. FastAPI 실행 (포트 8000)
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]