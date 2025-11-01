import sys
import json
import traceback
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from course.course_bundler import generate_course_for_topic

def safe_generate(topic: str):
    try:
        print(f"\n=== [ {topic} ] 코스 생성 시작 ===")
        generate_course_for_topic(topic)
        print(f"[ {topic} ] 코스 생성 완료\n")

    except json.JSONDecodeError as e:
        print(f"[ {topic} ] JSON 파싱 실패:")
        print(f"  └─ {e}")
        print("원인 추정:")
        print("    - 모델 출력에 따옴표 누락, 콜론(:) 오류, 중괄호 불일치 가능성")
        print("    - 너무 긴 입력(headlines > 5)으로 인해 출력 잘림 가능성")
        print("    - response_format 적용이 안 되었을 가능성")
        print("해결 팁: headlines 수 줄이기, 'JSON만 출력' 문구 추가\n")

    except Exception as e:
        print(f"[ {topic} ] 코스 생성 중 일반 오류 발생:")
        print(f"  └─ {type(e).__name__}: {e}")
        traceback.print_exc()
        print("원인 예시:")
        print("    - OpenAI API 응답 없음")
        print("    - 잘못된 키 접근 (예: resp.choices[0])")
        print("    - 파일 쓰기 중 경로 문제\n")

if __name__ == "__main__":
    safe_generate("tech")