from fastapi import FastAPI
from .endpoint.course_api import router as course_router
from pipeline.pipeline import run_learning_pipeline
from datetime import datetime
import asyncio
import pytz
import logging

app = FastAPI(title="NIEDU AI Server")

# 라우터 등록
app.include_router(course_router)

KST = pytz.timezone("Asia/Seoul")
logger = logging.getLogger(__name__)


async def daily_pipeline_scheduler():
    """한국시간 자정마다 자동 실행"""
    while True:
        now = datetime.now(KST)
        # 자정(00:00)에 한 번 실행
        if now.hour == 0 and now.minute == 0:
            logger.info("⏰ 자정 감지 → 자동 파이프라인 실행 시작")
            try:
                run_learning_pipeline()
                logger.info("✅ 파이프라인 자동 실행 완료")
            except Exception as e:
                logger.error(f"❌ 파이프라인 실행 실패: {e}")
            # 1분간 중복 실행 방지
            await asyncio.sleep(60)
        await asyncio.sleep(30)

@app.on_event("startup")
async def on_startup():
    """서버 기동 시 자동 스케줄러 실행"""
    asyncio.create_task(daily_pipeline_scheduler())
    logger.info("FastAPI 서버 기동 완료 - 파이프라인 자동 스케줄러 활성화")