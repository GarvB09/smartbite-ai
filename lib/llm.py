import json
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
MODEL = os.getenv("MODEL", "openai/gpt-oss-120b")


def get_llm():
    return ChatGroq(model=MODEL, temperature=0, max_tokens=1024)


def text_of(msg):
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") for b in c if isinstance(b, dict))


def parse_json(text):
    try:
        return json.loads(text[text.index("{") : text.rindex("}") + 1])
    except ValueError:
        return {"raw": text}