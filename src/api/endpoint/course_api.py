from fastapi import APIRouter
from pipeline.pipeline import run_learning_pipeline
from datetime import datetime
from pathlib import Path
import json
from starlette.responses import JSONResponse
from http.client import HTTPException

router = APIRouter(prefix="/api/course", tags=["Course"])

@router.get("/test")
def get_test_data():
    # 프로젝트 기준 상대 경로 설정
    file_path = Path("data/quiz/package/economy_2025-11-06_package.json")

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
    today = datetime.now().strftime("%Y-%m-%d")
    base_dir = Path(__file__).resolve().parents[3]
    quiz_package_dir = base_dir / "data" / "quiz" / "package"

    # 오늘 날짜의 통합 패키지 파일 검색
    package_files = list(quiz_package_dir.glob(f"*_{today}_package.json"))
    if not package_files:
        return {"message": f"{today} 기준 패키지 파일이 없습니다."}

    response = {}
    for f in package_files:
        topic = f.stem.split("_")[0]
        with open(f, "r", encoding="utf-8") as fp:
            response[topic] = json.load(fp)

    return response