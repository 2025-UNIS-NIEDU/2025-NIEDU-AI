# src/api/routes/course.py
from fastapi import APIRouter
from datetime import datetime
from src.api.models.course import CourseListResponse, Course, Topic, SubTopic, SubTag, Session, NewsRef, Step

router = APIRouter()

@router.get("/", response_model=CourseListResponse)
def get_course_list():
    return CourseListResponse(
        courses=[
            Course(
                title="정치적 논란 : 책임과 투명성의 재조명",
                shortDescription="정치권의 다양한 논란과 사퇴 사건을 중심으로...",
                longDescription="책임과 투명성의 중요성을 탐구하며...",
                topic=Topic(name="Politics"),
                subTags=[SubTag(name="정치적 책임"), SubTag(name="투명성 강화")],
                sessions=[
                    Session(
                        newsRef=NewsRef(
                            headline="與 백승아 '건진법사, 김 여사에 목걸이 전달 증언…특검 수사해야'",
                            publisher="연합뉴스",
                            topic="politics",
                            publishedAt=datetime.now(),
                            sourceUrl="https://news.naver.com"
                        ),
                        steps=[
                            Step(stepOrder=1, type="multi", content={"question": "..."})
                        ]
                    )
                ]
            )
        ]
    )