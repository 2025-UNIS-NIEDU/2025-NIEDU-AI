from fastapi import APIRouter
from wrapper.course_wrapper import build_course_package

router = APIRouter(prefix="/api/course", tags=["Course"])

@router.post("/build")
def build_course(topic: str):
    output_path = build_course_package(topic)
    return {"message": f"{topic} 통합 패키지 생성 완료", "path": str(output_path)}