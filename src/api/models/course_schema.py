from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Union, Any
from datetime import datetime


# === 기본 단위 ===
class SubTag(BaseModel):
    name: str = Field(..., description="소주제 해시태그명 (예: #기술혁신)")


class Topic(BaseModel):
    name: str = Field(..., description="상위 주제명 (예: 경제, 정치, 사회 등)")


class SubTopic(BaseModel):
    name: str = Field(..., description="하위 주제명 (예: 금융, 외교, 노동 등)")


# === 뉴스 관련 ===
class NewsRef(BaseModel):
    headline: str = Field(..., description="기사 제목")
    publisher: Optional[str] = Field(None, description="언론사명")
    publishedAt: Optional[datetime] = Field(None, description="기사 발행 시각")
    sourceUrl: Optional[HttpUrl] = Field(None, description="원문 링크 URL")
    thumbnailUrl: Optional[HttpUrl] = Field(None, description="기사 썸네일 이미지 URL")


# === 학습 단계 ===
class Step(BaseModel):
    stepOrder: int = Field(..., description="단계 순서 (1~5)")
    contentType: str = Field(..., description="콘텐츠 유형 (예: SUMMARY_READING, MULTIPLE_CHOICE 등)")
    contents: Union[List[Any], dict] = Field(..., description="실제 콘텐츠 데이터 (리스트 혹은 딕셔너리)")


class QuizLevel(BaseModel):
    level: str = Field(..., description="난이도 레벨 (N/I/E)")
    steps: List[Step] = Field(..., description="각 레벨 내 단계별 콘텐츠")


# === 세션 ===
class Session(BaseModel):
    sessionId: int = Field(..., description="세션 고유 ID")
    headline: str = Field(..., description="세션 대표 기사 제목")
    publishedAt: Optional[datetime] = Field(None, description="대표 기사 발행 시각")
    publisher: Optional[str] = Field(None, description="대표 기사 언론사명")
    sourceUrl: Optional[HttpUrl] = Field(None, description="대표 기사 URL")
    summary: Optional[str] = Field(None, description="요약문 (요약문 기반 퀴즈 생성에 사용)")
    thumbnailUrl: Optional[HttpUrl] = Field(None, description="썸네일 이미지 URL")
    quizzes: List[QuizLevel] = Field(..., description="세션 내 퀴즈 단계별 데이터")


# === 코스 ===
class Course(BaseModel):
    courseId: int = Field(..., description="코스 ID")
    topic: str = Field(..., description="상위 주제명 (예: 경제, 정치 등)")
    subTopic: str = Field(..., description="하위 주제명 (예: 금융, 노동 등)")
    subTags: List[str] = Field(..., description="코스 소주제 해시태그 리스트")
    courseName: str = Field(..., description="코스명 (예: 금융시장 변화 : 정책 전환의 흐름)")
    courseDescription: str = Field(..., description="코스 설명문")
    sessions: List[Session] = Field(..., description="세션 목록")


# === 전체 래퍼 ===
class CourseWrapper(BaseModel):
    courses: List[Course] = Field(..., description="전체 코스 패키지 데이터")