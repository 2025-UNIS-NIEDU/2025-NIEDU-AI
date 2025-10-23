from fastapi import FastAPI

# FastAPI 앱 생성
app = FastAPI()

# 서버가 살아있는지 확인하는 기본 API (Health Check)
@app.get("/ai/health")
def health_check():
    return {"status": "AI server is running!"}


# 여기에 AI 모델을 로딩하고,실제 예측을 수행하는 API 엔드포인트들을 추가
