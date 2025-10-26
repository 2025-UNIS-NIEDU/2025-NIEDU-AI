# src/api/models/course.py
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class Topic(BaseModel):
    name: str

class SubTopic(BaseModel):
    name: str

class NewsRef(BaseModel):
    headline: str
    publisher: Optional[str] = None
    topic: Optional[str] = None
    publishedAt: datetime
    thumbnailUrl: Optional[str] = None
    sourceUrl: Optional[str] = None

class Step(BaseModel):
    stepOrder: int
    type: str
    content: dict

class Session(BaseModel):
    newsRef: NewsRef
    steps: List[Step]

class Course(BaseModel):
    title: str
    shortDescription: str
    longDescription: str
    thumbnailUrl: Optional[str] = None
    topic: Topic
    subTopic : SubTopic
    subTags: List[SubTag]
    sessions: List[Session]

class CourseListResponse(BaseModel):
    courses: List[Course]