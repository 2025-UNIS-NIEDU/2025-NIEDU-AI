# clean_data.py
import os
from pathlib import Path
from datetime import datetime

def clean_old_files(base_path: Path):
    """
    data 폴더 내부의 모든 파일 중
    오늘 날짜(YYYY-MM-DD)가 파일명에 포함되지 않으면 삭제.
    """
    today = datetime.now().strftime("%Y-%m-%d")

    for root, dirs, files in os.walk(base_path):
        for file in files:
            file_path = Path(root) / file

            # 오늘 날짜 포함 X → 삭제
            if today not in file:
                try:
                    file_path.unlink()
                except Exception as e:
                    print(f"삭제 실패: {file_path} | {e}")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[1] / "data"
    clean_old_files(BASE_DIR)