import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
# Groq speaks the OpenAI protocol. Only the base_url changes.
client = OpenAI(
 api_key=os.environ["GROQ_API_KEY"],
 base_url="https://api.groq.com/openai/v1",
)
MODEL = "openai/gpt-oss-20b"
r = client.chat.completions.create(
 model=MODEL,
 messages=[{"role": "user", "content": "Reply with exactly: OK"}],
)
choice = r.choices[0]
print("content :", choice.message.content)
print("finish_reason :", choice.finish_reason)
print("input tokens :", r.usage.prompt_tokens)
print("output tokens :", r.usage.completion_tokens)