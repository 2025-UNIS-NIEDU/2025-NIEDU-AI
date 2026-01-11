from fastapi import APIRouter, HTTPException
from src.pipeline.pipeline import run_learning_pipeline
from datetime import datetime
from pathlib import Path
import json
import logging
from starlette.responses import JSONResponse

router = APIRouter(prefix="/api/course", tags=["Course"])
logger = logging.getLogger(__name__)

@router.get("/test")
def get_test_data():
    # 프로젝트 기준 상대 경로 설정
    file_path = Path("data/quiz/package/economy_2025-11-24_package.json")
    logger.info("get_test_data file_path=%s", file_path)

    # 파일 존재 여부 체크
    if not file_path.exists():
        logger.warning("get_test_data file not found: %s", file_path)
        raise HTTPException(status_code=404, detail="JSON file not found")

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("get_test_data loaded JSON keys=%s", list(data.keys()) if isinstance(data, dict) else type(data))
        return JSONResponse(content=data)
    except json.JSONDecodeError:
        logger.exception("get_test_data invalid JSON format: %s", file_path)
        raise HTTPException(status_code=500, detail="Invalid JSON format")
    except Exception as e:
        logger.exception("get_test_data unexpected error: %s", file_path)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.get("/today")
def get_today_data():
    """오늘 날짜 기준 통합 패키지 JSON 리턴"""
    base_dir = Path(__file__).resolve().parents[3]
    quiz_package_dir = base_dir / "data" / "quiz" / "package"

    today = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
    package_files = sorted(list(quiz_package_dir.glob(f"*_{today}_package.json")))
    logger.info(
        "get_today_data base_dir=%s quiz_package_dir=%s today=%s file_count=%d",
        base_dir,
        quiz_package_dir,
        today,
        len(package_files),
    )
    if package_files:
        logger.info(
            "get_today_data files=%s",
            [str(p) for p in package_files],
        )

    # 1. 파일이 없을 경우: 백엔드 형식에 맞게 {"courses": []} 반환
    if not package_files:
        logger.warning("get_today_data no package files found")
        return {"courses": []}

    # 2. 모든 패키지를 하나의 리스트로 통합
    all_courses = []

    for f in package_files:
        logger.info("get_today_data reading file=%s", f)
        with open(f, "r", encoding="utf-8") as fp:
            course_data = json.load(fp)
            if isinstance(course_data, list):
                logger.info("get_today_data file=%s list_len=%d", f, len(course_data))
                all_courses.extend(course_data)
            else:
                logger.info("get_today_data file=%s type=%s", f, type(course_data))
                all_courses.append(course_data)

                # 3. AICourseListResponse 형식으로 래핑하여 반환
    logger.info("get_today_data total_courses=%d", len(all_courses))
    return {"courses": all_courses}


@router.get("/packages/all")
def get_all_packages():
    """data/quiz/package 내 모든 JSON을 통합해 courses 리스트로 반환"""
    base_dir = Path(__file__).resolve().parents[3]
    quiz_package_dir = base_dir / "data" / "quiz" / "package"

    package_files = sorted(list(quiz_package_dir.glob("*.json")))
    logger.info(
        "get_all_packages base_dir=%s quiz_package_dir=%s file_count=%d",
        base_dir,
        quiz_package_dir,
        len(package_files),
    )
    if not package_files:
        logger.warning("get_all_packages no package files found")
        return {"courses": []}

    all_courses = []
    for f in package_files:
        logger.info("get_all_packages reading file=%s", f)
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
        except json.JSONDecodeError:
            logger.exception("get_all_packages invalid JSON format: %s", f.name)
            raise HTTPException(status_code=500, detail=f"Invalid JSON format: {f.name}")
        except Exception as e:
            logger.exception("get_all_packages unexpected error: %s", f.name)
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

        if isinstance(data, dict) and isinstance(data.get("courses"), list):
            logger.info("get_all_packages file=%s courses_len=%d", f, len(data["courses"]))
            all_courses.extend(data["courses"])
        elif isinstance(data, list):
            logger.info("get_all_packages file=%s list_len=%d", f, len(data))
            all_courses.extend(data)
        else:
            logger.info("get_all_packages file=%s type=%s", f, type(data))
            all_courses.append(data)

    logger.info("get_all_packages total_courses=%d", len(all_courses))
    return {"courses": all_courses}
