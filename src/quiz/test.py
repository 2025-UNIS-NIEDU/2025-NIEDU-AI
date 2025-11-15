import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # src
sys.path.append(str(Path(__file__).resolve().parents[2]))  # github
from quiz.multi import generate_multi_choice_quiz
from quiz.short import generate_short_quiz

SESSIONS = [
    {
        "courseId": "1",
        "sessionId": "1",
        "courseName": "한-싱가포르 협력 : 외교적 전환",
        "topic": "politics",
        "headline": "한-싱가포르 정상회담...'전략적 동반자 관계' 수립",
        "sourceUrl": "https://www.ytn.co.kr/_ln/0101_202511022323314491",
        "summary": "2023년 10월 2일, 한국의 이재명 대통령과 싱가포르의 로렌스 웡 총리가 용산 대통령실에서 정상회담을 열었다. 이 자리에서 이재명 대통령은 두 나라가 전략적 동반자 관계를 수립했다고 발표했다. 로렌스 웡 총리는 올해가 양국의 수교 50주년인 점을 강조하며, 정상회담을 통해 양국 관계의 우수성을 점검하고 미래의 발전 가능성을 확인하여 외교 관계를 격상하게 되었다고 설명하였다."
    },
    {
        "courseId": "1",
        "sessionId": "2",
        "topic": "politics",
        "headline": "대장동 1심 판결 나오자… 與 ‘재판중지법’ 공식화",
        "sourceUrl": "https://www.kmib.co.kr/article/view.asp?arcid=1762071026&code=11121100",
        "summary": "더불어민주당이 현직 대통령의 형사 사건 재판을 멈추게 하는 ‘재판중지법’(형사소송법 개정안) 추진을 공식화했다. 민주당은 재판중지법 처리를 통해 ‘정치보복성 기소’를 방지하겠다는 입장을 밝혔다."
    },
    {
        "courseId": "1",
        "sessionId": "3",
        "topic": "politics",
        "headline": "정청래 \"지선, '제2내란극복' 선거될 것…민주적 경선이 승리 주춧돌\"",
        "publishedAt": "2025-11-02T18:05:00",
        "publisher": "더팩트",
        "sourceUrl": "https://news.tf.co.kr/read/ptoday/2258245.htm",
        "summary": "\"예비후보 부적격자 아니면 누구라도 경선에 참여하고 승복하는 선례 만들 것\" 정청래 더불어민주당 대표가 \"대통령선거는 제1의 내란극복, 지방선거는 내란잔재를 청산하는 제2의 내란극복 선거\"라고 밝혔다.\n\n그는 \"가장 민주적인 경선이 가장 큰 승리를 가져오는 주춧돌이 될 것\"이라며 \"예비후보 부적격자가 아니면 누구라도 경선에 참여하고 승복하는 선례를 만들기 위해 당대표로서 최선을 다하겠다\"고 다짐했다.\n\n한편 정 대표는 이날 오전 전남 나주종합스포츠파크 다목적체육관에서 열린 전남도당 임시 당원대회에서 \"지금까지와는 완전히 다른 100% 당원이 주인 되는 경선, 당원들의 마음이 100% 녹아서 관철되는 완전한 민주적인 경선\"을 예고했다."
    },
    { 
        "courseId": "1",  
        "sessionId": "4",
        "topic": "economy",
        "headline": "서울 아파트 매수 심리 꺾였다… 강남권보다 강북권서 더 위축",
        "sourceUrl": "https://www.seoul.co.kr/news/economy/estate/2025/11/03/20251103002001",
        "summary": "‘10·15 대책’ 여파로 9주 만에 하락 대출 규제 탓 강북 실수요층 타격 서울 아파트 낙찰가율 100% 돌파 서울을 비롯한 경기 12곳을 ‘규제지역’으로 묶은 10·15 부동산 대책 이후 아파트 매수 심리가 꺾인 것으로 나타났다.\n\n2일 한국부동산원 주간 아파트 수급 동향에 따르면 10월 넷째 주(10월 27일 기준) 매매수급지수는 서울 아파트의 경우 직전 주(105.4) 대비 2.2포인트 내린 103.2를 기록했다.\n\n지난 6월까지 가파르게 오르던 서울 아파트 매매수급지수는 고강도 대출 규제를 담은 6·27 대책 시행 이후 한때 100 밑으로 떨어졌다."
    },
]

if __name__ == "__main__":
    # 특정 세션만 선택해서 테스트할 때
    for sess in SESSIONS:
        try:
            print(f"🚀 {sess['headline']}  → 퀴즈 생성 시작")
            generate_short_quiz(selected_session=sess)
            print(f"✅ {sess['headline']}  → 완료\n")
        except Exception as e:
            print(f"❌ {sess['headline']}  → 오류 발생: {e}\n")