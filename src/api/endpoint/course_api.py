from fastapi import APIRouter
from src.pipeline.pipeline import run_learning_pipeline
from datetime import datetime
from pathlib import Path
import json

router = APIRouter(prefix="/api/course", tags=["Course"])

@router.post("/build")
def build_course():
    """전체 학습 파이프라인 수동 실행"""
    result = run_learning_pipeline()
    return {"message": "전체 파이프라인 실행 완료", "data": result}

@router.get("/today")
def get_today_data():
    """오늘 날짜 기준 통합 패키지 JSON 리턴"""
    base_dir = Path(__file__).resolve().parents[3]
    quiz_package_dir = base_dir / "data" / "quiz" / "package"

    # 1. 모든 패키지 파일을 다 찾기
    package_files = sorted(list(quiz_package_dir.glob("*_package.json")))

    if not package_files:
        # 파일이 하나도 없으면 오늘 날짜로 메시지 리턴
        today = datetime.now().strftime("%Y-%m-%d")
        return {"message": f"{today} 기준 패키지 파일이 없습니다."}

    # 2. 가장 최신 파일 하나를 가져옴
    latest_file = package_files[-1]

    # 3. 그 파일을 열어서 리턴
    response = {}

    for f in package_files:
        topic = f.stem.split("_")[0]
        with open(f, "r", encoding="utf-8") as fp:
            response[topic] = json.load(fp)

    return response