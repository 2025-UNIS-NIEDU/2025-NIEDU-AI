from fastapi import APIRouter, HTTPException
from src.pipeline.pipeline import run_learning_pipeline
from datetime import datetime
from pathlib import Path
import json
from starlette.responses import JSONResponse

router = APIRouter(prefix="/api/course", tags=["Course"])

@router.get("/test")
def get_test_data():
    # 프로젝트 기준 상대 경로 설정
    file_path = Path("data/quiz/package/economy_2025-11-24_package.json")

    # 파일 존재 여부 체크
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="JSON file not found")

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.get("/today")
def get_today_data():
    """오늘 날짜 기준 통합 패키지 JSON 리턴"""
    base_dir = Path(__file__).resolve().parents[3]
    quiz_package_dir = base_dir / "data" / "quiz" / "package"

    package_files = sorted(list(quiz_package_dir.glob("*_package.json")))

    # 1. 파일이 없을 경우: 백엔드 형식에 맞게 {"courses": []} 반환
    if not package_files:
        return {"courses": []}

    # 2. 모든 패키지를 하나의 리스트로 통합
    all_courses = []

    for f in package_files:
        with open(f, "r", encoding="utf-8") as fp:
            course_data = json.load(fp)
            if isinstance(course_data, list):
                all_courses.extend(course_data)
            else:
                all_courses.append(course_data)

                # 3. AICourseListResponse 형식으로 래핑하여 반환
    return {"courses": all_courses}


@router.get("/packages/all")
def get_all_packages():
    """data/quiz/package 내 모든 JSON을 통합해 courses 리스트로 반환"""
    base_dir = Path(__file__).resolve().parents[3]
    quiz_package_dir = base_dir / "data" / "quiz" / "package"

    package_files = sorted(list(quiz_package_dir.glob("*.json")))
    if not package_files:
        return {"courses": []}

    all_courses = []
    for f in package_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail=f"Invalid JSON format: {f.name}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

        if isinstance(data, dict) and isinstance(data.get("courses"), list):
            all_courses.extend(data["courses"])
        elif isinstance(data, list):
            all_courses.extend(data)
        else:
            all_courses.append(data)

    return {"courses": all_courses}
