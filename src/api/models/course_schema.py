from pydantic import BaseModel, Field
from typing import List, Optional, Any, Union
from datetime import datetime

class SubTag(BaseModel):
    name: str = Field(..., description="소주제 해시태그명 (예: 정치개혁)")

class Topic(BaseModel):
    name: str = Field(..., description="상위 주제명 (예: Politics)")

class SubTopic(BaseModel):
    name: str = Field(..., description="하위 주제명 (예: 정당)")

class NewsRef(BaseModel):
    headline: str
    publisher: Optional[str] = None
    publishedAt: Optional[datetime] = None
    sourceUrl: Optional[str] = None
    thumbnailUrl: Optional[str] = None

class Step(BaseModel):
    stepOrder: int
    contentType: str
    contents: Union[List[Any], dict]  # 복수 문항 또는 단일 질문

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

class CourseWrapper(BaseModel):
    courses: List[Course]