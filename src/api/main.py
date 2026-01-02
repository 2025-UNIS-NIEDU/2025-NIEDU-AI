from fastapi import FastAPI
from src.api.endpoint.course_api import router as course_router
from src.pipeline.pipeline import run_learning_pipeline
from datetime import datetime
import asyncio
import pytz
import logging
import os
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED

app = FastAPI(title="NIEDU AI Server")

# 라우터 등록
app.include_router(course_router)

KST = pytz.timezone("Asia/Seoul")
logger = logging.getLogger(__name__)

scheduler: AsyncIOScheduler | None = None

def _get_schedule_time():
    """환경 변수로 스케줄 시간 지정 (기본: KST 22:30)."""
    hour = int(os.getenv("PIPELINE_HOUR", "22"))
    minute = int(os.getenv("PIPELINE_MINUTE", "30"))
    return hour, minute

def _scheduler_listener(event):
    if event.exception:
        logger.error("❌ 스케줄 잡 실패", exc_info=event.exception)
    else:
        logger.info("✅ 스케줄 잡 완료")

async def _run_pipeline_job():
    logger.info("⏰ 스케줄 잡 시작 → 자동 파이프라인 실행")
    try:
        await asyncio.to_thread(run_learning_pipeline)
    except Exception:
        logger.exception("❌ 파이프라인 실행 실패")
        raise

@app.on_event("startup")
async def on_startup():
    """서버 기동 시 자동 스케줄러 실행"""
    global scheduler
    if os.getenv("PIPELINE_SCHEDULER_ENABLED", "1") != "1":
        logger.warning("파이프라인 스케줄러 비활성화됨 (PIPELINE_SCHEDULER_ENABLED!=1)")
        return

    if scheduler and scheduler.running:
        logger.warning("스케줄러가 이미 실행 중입니다.")
        return

    hour, minute = _get_schedule_time()
    trigger = CronTrigger(hour=hour, minute=minute, timezone=KST)

    scheduler = AsyncIOScheduler(timezone=KST)
    scheduler.add_listener(_scheduler_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
    scheduler.add_job(
        _run_pipeline_job,
        trigger=trigger,
        id="daily_learning_pipeline",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )
    scheduler.start()

    next_run = scheduler.get_job("daily_learning_pipeline").next_run_time
    logger.info("FastAPI 서버 기동 완료 - 파이프라인 스케줄러 활성화")
    logger.info(f"스케줄 시간: KST {hour:02d}:{minute:02d}")
    logger.info(f"다음 실행 예정: {next_run}")

@app.on_event("shutdown")
async def on_shutdown():
    """서버 종료 시 스케줄러 정리"""
    global scheduler
    if scheduler:
        logger.info("스케줄러 종료")
        scheduler.shutdown(wait=False)
