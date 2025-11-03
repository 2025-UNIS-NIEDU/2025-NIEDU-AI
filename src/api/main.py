# src/api/main.py
from fastapi import FastAPI
from api.endpoint.course_api import router as course_router

app = FastAPI()

# 라우터 등록
app.include_router(course_router)