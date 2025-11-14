import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "이 이미지의 주요 장면과 분위기를 설명해줘."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://ddi-cdn.deepsearch.com/news_thumbnail/world/2025/11/02/1636503061050560961/000-534bec1596bafe3288d26623a9fc0fe8f4700a24.jpg"
                    },
                },
            ],
        }
    ],
)

print(response.choices[0].message.content)