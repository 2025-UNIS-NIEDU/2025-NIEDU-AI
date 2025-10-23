import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from course.course_bundler import generate_course_for_topic

if __name__ == "__main__":
    generate_course_for_topic("world")