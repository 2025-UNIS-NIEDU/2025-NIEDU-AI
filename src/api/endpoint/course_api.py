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