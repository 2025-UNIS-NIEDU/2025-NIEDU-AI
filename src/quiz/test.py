import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # src
sys.path.append(str(Path(__file__).resolve().parents[2]))  # github
from quiz.multi import generate_multi_choice_quiz
from quiz.short import generate_short_quiz
from src.quiz.summary_reading import generate_summary_reading_quiz

SESSIONS = [
    {
        "courseId" : "1",
        "sessionId": "1",
        "topic" : "politics",
        "headline": "한동훈 “조국, 토론 피하지 말고 사면 밥값해라”",
        "publishedAt": "2025-11-15T21:59:00",
        "publisher": "문화일보",
        "sourceUrl": "https://www.munhwa.com/article/11547085",
        "summary": "“토론 제의, 조국 헛소리에서 시작” 조국 조국혁신당 비상대책위원장과 한동훈 전 국민의힘 대표.\n\n연합뉴스 뉴시스 한동훈 전 국민의힘 대표는 한 전 대표가 제안한 토론에 응하지 않겠다 선을 그은 조국 조국혁신당 비상대책위원장을 향해 “도망가지 말고 ‘특혜사면’ 밥값해라”고 비꼬았다.\n\n단, 이하는 말한다‘면서 근엄하게 딴소리하며 도망가려 하는데, 토론에서 조국 씨 떠들고 싶은 대로 주제 제한 없이 다 받아준다”면서 “울지 말고 얘기하라”고 덧붙였다.",
        "thumbnailUrl": "https://ddi-cdn.deepsearch.com/news_thumbnail/politics/2025/11/15/1641304151793209728/000-1ac483755d60444b392cc7be89dbe5c227000280.jpg"
      }
]

if __name__ == "__main__":
    for sess in SESSIONS:
        try:
            print(f"{sess['headline']}  → 퀴즈 생성 시작")
            generate_summary_reading_quiz(selected_session=sess)
            print(f"{sess['headline']}  → 완료\n")
        except Exception as e:
            print(f"{sess['headline']}  → 오류 발생: {e}\n")