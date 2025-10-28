from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime


# === 하위 구조 정의 ===

class SubTag(BaseModel):
    name: str


class Topic(BaseModel):
    name: str


class SubTopic(BaseModel):
    name: str


class NewsRef(BaseModel):
    headline: str
    publisher: Optional[str] = None
    publishedAt: Optional[str] = None  # ISO 문자열
    sourceUrl: Optional[str] = None
    thumbnailUrl: Optional[str] = None


class Step(BaseModel):
    stepOrder: int
    contentType: str
    contents: Any  # dict or list — contentType마다 달라서 Any로 처리


class QuizLevel(BaseModel):
    level: str
    steps: List[Step]


class Session(BaseModel):
    newsRef: NewsRef
    quizzes: List[QuizLevel]


class Course(BaseModel):
    courseName: str
    topic: Topic
    subTopic: SubTopic
    subTags: List[SubTag]
    sessions: List[Session]


# === 최상위 래퍼 ===
class CourseWrapper(BaseModel):
    courses: List[Course]