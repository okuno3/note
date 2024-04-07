import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "動作テスト。動作するなら「はい」と回答"},
    ],
)
print(response.choices[0]["message"]["content"])