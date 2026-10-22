"""Chat with the agent in the terminal: python cli.py
Needs OPENAI_API_KEY (and optionally OPENAI_BASE_URL, MODEL) in the environment or .env."""
import os

from dotenv import load_dotenv

from agent import build_agent, openai_model

load_dotenv()
agent = build_agent(openai_model(os.getenv("MODEL", "gpt-4o-mini")), log=lambda s: print("   " + s))
print("Пишите заявку или вопрос (пустая строка — выход).")
while True:
    text = input("> ").strip()
    if not text:
        break
    print(agent.reply("cli-user", text))
