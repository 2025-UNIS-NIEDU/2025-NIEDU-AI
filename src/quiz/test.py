import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # src
sys.path.append(str(Path(__file__).resolve().parents[2]))  # github
from quiz.short import generate_short_quiz
from src.quiz.summary_reading import generate_summary_reading_quiz
from src.quiz.current_affairs import generate_current_affairs_quiz
from quiz.term import generate_term_quiz

SESSIONS = [
      {
        "courseId" : "1",
        "sessionId": "2",
        "topic" : "society",
        "headline": "한강버스 저수심·이물질 걸림 15번…\"정밀 조사로 안전성 확보\"",
        "publishedAt": "2025-11-17T12:17:00",
        "publisher": "SBS",
        "sourceUrl": "https://news.sbs.co.kr/news/endPage.do?news_id=N1008333968",
        "summary": "한강버스가 항로를 이탈해 수심이 얕은 강바닥에 걸려 멈추는 사고가 발생한 가운데 한강버스가 정식 항로를 운항하던 중에도 강바닥이나 이물질 등에 닿았다는 보고가 총 15차례 나왔던 것으로 나타났습니다.\n\n이번 사고와 별개로 앞서 서울시와 ㈜한강버스는 뚝섬 선착장 부근이 수심이 낮다는 점을 고려해 16일부터 28일까지 이곳을 무정차 통과하고 이물질과 부유물질을 제거하는 작업에 착수하기로 했습니다.\n\n이어 15일 오후 8시 24분 한강버스가 잠실 선착장 인근에서 항로를 이탈, 저수심 구간으로 진입해 강바닥에 걸려 멈추는 사고가 발생했습니다.",
        "thumbnailUrl": "https://ddi-cdn.deepsearch.com/news_thumbnail/society/2025/11/17/1641882642683138610/000-198d8b284e13ffed5b411b2a16ff4f4c82b60e02.jpg"
      },
      {
        "courseId" : "1",
        "sessionId": "3",
        "topic" : "society",
        "headline": "한강버스 '터치' 보고 이미 15건‥압구정~잠실 구간 무기한 운항 중단",
        "publishedAt": "2025-11-17T12:09:00",
        "publisher": "MBC",
        "sourceUrl": "https://imnews.imbc.com/news/2025/society/article/6776198_36718.html",
        "summary": "한강버스가 항로를 이탈해 운항하다 강바닥에 걸려 멈춘 사고가 발생한 가운데, 이와 유사하게 한강버스가 강바닥에 걸리거나 부유물에 부딪히는 등 관련 보고가 총 15건 있었던 사실이 뒤늦게 드러났습니다.\n\n김선직 주식회사 한강버스 대표는 오늘 오전 서울시청에서 열린 브리핑에서 수심이 낮거나 이물질에 부딪히는 등 '터치' 보고가 몇 회 있었냐는 질문에 \"15회 정도 들어왔다\"고 답했습니다.\n\n지난 15일 잠실선착장 인근에서 발생한 멈춤사고에 대해서는 \"선박이 지정항로를 이탈해 저수심 구간에 걸린 것\"이라며 \"선장은 오른쪽 항로표시등이 잘 보이지 않았다고 진술했다\"고 전했습니다.",
        "thumbnailUrl": "https://ddi-cdn.deepsearch.com/news_thumbnail/society/2025/11/17/1641881127285297217/000-843ea7d7b414a83a81b7b5a930d5da7c4c280adc.jpg"
      },
      {
        "courseId" : "1",
        "sessionId": "4",
        "topic" : "society",
        "headline": "중국 해경, 서해해경청에 감사 서한 전달 \"헌신적 수색구조 감사\"",
        "publishedAt": "2025-11-17T12:19:00",
        "publisher": "뉴스1",
        "sourceUrl": "https://www.news1.kr/local/gwangju-jeonnam/5978354",
        "summary": "17일 서해해경청에 따르면 지난 13일 중국 해경국 북해분국으로부터 한국 해양경찰의 신속하고 적극적인 수색구조 활동에 대한 '감사 서한문'이 왔다.\n\n중국 해경은 서한문을 통해 \"한국 해경이 자국(중국 해경)과 긴밀히 협력하고 함께 힘을 모아 효율적인 수색 구조를 진행한 데 대해 특별한 감사를 보낸다\"며 \"앞으로 양 기관이 해양질서 수호를 위해 함께 노력해 나가길 기대한다\"고 전했다.\n\n지난 11일에는 중국 주 광주총영사관 주적화 부 총영사가 서해해경청을 방문해 한국 해양경찰의 헌신적인 수색구조 활동에 중국 정부와 선원 가족을 대신해 깊은 감사를 표하기도 했다.",
        "thumbnailUrl": "https://ddi-cdn.deepsearch.com/news_thumbnail/society/2025/11/17/1641883162390958686/000-496b6e0214c212bdc3868ddbdf8a867ca6dc8a3a.jpg"
      },
]

if __name__ == "__main__":
    for sess in SESSIONS:
        try:
            print(f"{sess['headline']}  → 퀴즈 생성 시작")
            generate_short_quiz(selected_session=sess)
            print(f"{sess['headline']}  → 완료\n")
        except Exception as e:
            print(f"{sess['headline']}  → 오류 발생: {e}\n")