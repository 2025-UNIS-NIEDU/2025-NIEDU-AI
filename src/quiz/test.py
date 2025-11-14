import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # src
sys.path.append(str(Path(__file__).resolve().parents[2]))  # github
from quiz.summary_reading import generate_summary_reading_quiz
from quiz.multi import generate_multi_choice_quiz
from quiz.short import generate_short_quiz

SESSIONS = [
    {
        "courseId" : "1",
        "sessionId": "1",
        "topic" : "politics",
        "headline": "한-중, 사드 9년만에 “관계 전면 복원”…북핵·한한령 등 과제 남아",
        "publishedAt": "2025-11-02T19:14:00",
        "publisher": "한겨레",
        "sourceUrl": "https://www.hani.co.kr/arti/politics/politics_general/1226960.html",
        "summary": "이 대통령이 회담을 앞두고 계속 강조한 한반도 비핵화를 위한 중국의 역할에 대해서는 원론적 입장을 확인하는 데 그쳤다.\n\n이 대통령은 이날 정상회담 머리발언에서 “한반도 평화와 안정을 정착시키는 데도 중국의 역할은 매우 중요하다”며 “한반도가 안정돼야 동북아도 안정되고, 그것이 중국의 이익에도 부합할 것이다.\n\n이 대통령과 시 주석은 중국의 한화오션 제재, 서해 구조물 활동, 한한령 등 민감하고 껄끄러운 사안에 대해서도 의논했지만 구체적인 답을 도출하지는 못했다.",
        "thumbnailUrl": "https://ddi-cdn.deepsearch.com/news_thumbnail/politics/2025/11/02/1636551477092814906/000-ca853f63d4807b230654be3edc2687c76fd45337.jpg"
    },
    {
        "courseId" : "1",
        "sessionId": "2",
        "topic" : "society",
        "headline": "강릉시보건소,강릉영동대와 다모아 사업 4년째 펼쳐",
        "publishedAt": "2025-11-02T23:06:00",
        "publisher": "강원일보",
        "sourceUrl": "https://www.kwnews.co.kr/page/view/2025110213452346585",
        "summary": "강릉시보건소는 강릉영동대와 함께 의료취약지 어르신들의 건강생활 실천을 돕기 위한 ‘다모아 사업’을 4년째 이어간다.\n\n물리치료과, 안경광학과, 사회복지학과 등 교수진과 학생들이 직접 참여해 전문 보건 지식을 활용, 어르신들에게 근골격 운동법, 눈 건강 관리 교육 및 만들기 활동을 통한 미술치료 등 다양한 체험형 교육을 제공하고 있다.\n\n이를 통해 신체적 건강은 물론 정서적 안정과 사회적 교류를 함께 도모할 수 있도록 지원하며 지역사회 건강증진의 모범 프로그램으로 자리 잡고 있다.",
        "thumbnailUrl": "https://ddi-cdn.deepsearch.com/news_thumbnail/society/2025/11/02/1636609842955293064/000-5c99e39461fd1e37cbd4fdcdeddf62f4bd351a4c.jpeg"
    },
    {
        "courseId" : "1",
        "sessionId": "3",
        "topic" : "economy",
        "headline": "환율 당분간 1400원대 고공행진…연평균으로 외환위기 때 넘어설듯",
        "publishedAt": "2025-11-02T18:21:00",
        "publisher": "파이낸셜뉴스",
        "sourceUrl": "http://www.fnnews.com/news/202511021819531419",
        "summary": "미국 연방준비제도(연준)의 12월 금리 신중론에 달러화가 강세를 보이고 있어 올해 연평균 환율이 외환위기 이후 최고 수준을 기록할 가능성도 커졌다.\n\n임혜윤 한화투자증권 연구원은 \"대미투자가 늘어난다는 점은 변화가 없다\"며 \"우리가 들여올 수 있는 운용수익이 줄고, 국내에서 진행될 수 있는 투자 일부가 해외에서 이뤄지게 돼 중장기적으로 원·달러 환율 하락을 제한할 가능성이 있다\"고 짚었다.\n\n이에 올해 연평균 원·달러 환율이 역대 최고치였던 외환위기 직후를 넘을 가능성이 커졌다는 평가가 나온다.",
        "thumbnailUrl": "https://ddi-cdn.deepsearch.com/news_thumbnail/economy/2025/11/02/1636538532728279639/000-233a72e895d6a7a0338b3d0b30400f33eee48838.jpg"
    },
    {
        "courseId" : "1",
        "sessionId": "4",
        "topic" : "world",
        "headline": "미 “중, 조선·해운 보복조치 철회키로”",
        "publishedAt": "2025-11-02T19:02:00",
        "publisher": "KBS",
        "sourceUrl": "https://news.kbs.co.kr/news/pc/view/view.do?ncd=8396590&ref=A",
        "summary": "도널드 트럼프 미국 대통령과 시진핑 중국 국가주석이 타결한 무역 합의의 일환으로 중국이 한화오션의 미국 자회사에 부과한 제재를 철회할 가능성이 제기됩니다.\n\n미국 백악관이 현지시각 1일, 공개한 미중 정상 간 무역 합의 팩트시트에 따르면 중국은 미국의 '무역법 301조' 조사에 보복하기 위해 시행한 조치를 철회하고 다양한 해운 기업에 부과한 제재도 철회하기로 했습니다.\n\n앞서 중국은 지난 14일 한화오션의 미국 자회사 5곳을 중국 기업과 거래가 금지된 제재 목록에 올렸습니다.",
        "thumbnailUrl": "https://ddi-cdn.deepsearch.com/news_thumbnail/world/2025/11/02/1636551077425975809/000-ec56fcc0f6b951bc663fde7b12f6cbc63d4b7c34.jpg"
    },
]

if __name__ == "__main__":
    for sess in SESSIONS:
        try:
                print(f"{sess['headline']}  → 퀴즈 생성 시작")
                #generate_summary_reading_quiz(selected_session=sess)
                #generate_multi_choice_quiz(selected_session=sess)
                generate_short_quiz(selected_session=sess)
                print(f"{sess['headline']}  → 완료\n")
        except Exception as e:
            print(f"{sess['headline']}  → 오류 발생: {e}\n")