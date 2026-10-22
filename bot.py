"""Telegram bot front-end (aiogram 3). Run: python bot.py
Env: BOT_TOKEN, OPENAI_API_KEY, optional OPENAI_BASE_URL, MODEL, MANAGER_CHAT_ID."""
import asyncio
import logging
import os

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message
from dotenv import load_dotenv
from openai import OpenAI

from agent import build_agent, openai_model

load_dotenv()
logging.basicConfig(level=logging.INFO)
client = OpenAI()
agent = build_agent(openai_model(os.getenv("MODEL", "gpt-4o-mini"), client), log=logging.info)
bot = Bot(os.environ["BOT_TOKEN"])
dp = Dispatcher()
MANAGER = os.getenv("MANAGER_CHAT_ID")


@dp.message(CommandStart())
async def start(m: Message):
    await m.answer("Здравствуйте! Напишите заявку текстом или голосом, например: «2 ящика швепса и 3 кг лаймов на завтра». "
                   "Ещё могу ответить про доставку, оплату и возвраты.")


async def handle(m: Message, text: str):
    customer = f"{m.from_user.full_name} (tg:{m.from_user.id})"
    before = len(agent.crm._load())
    answer = await asyncio.to_thread(agent.reply, customer, text)
    await m.answer(answer)
    orders = agent.crm._load()
    if MANAGER and len(orders) > before:  # a new order was created during this turn
        o = orders[-1]
        lines = "\n".join(f"• {i['name']} — {i['qty']:g} {i['unit']}" for i in o["items"])
        await bot.send_message(MANAGER, f"Новая заявка {o['id']} от {o['customer']} на {o['delivery_date']}:\n{lines}\nИтого: {o['total']:.0f} ₽"
                               + (f"\n⚠ {'; '.join(o['warnings'])}" if o["warnings"] else ""))


@dp.message(F.voice)
async def voice(m: Message):
    # Voice → text via the Whisper-compatible transcription endpoint, then the same agent.
    file = await bot.download(m.voice.file_id)
    file.name = "voice.ogg"
    text = (await asyncio.to_thread(client.audio.transcriptions.create, model=os.getenv("STT_MODEL", "whisper-1"), file=file)).text
    await m.answer(f"🎤 {text}")
    await handle(m, text)


@dp.message(F.text)
async def text(m: Message):
    await handle(m, m.text)


if __name__ == "__main__":
    asyncio.run(dp.start_polling(bot))
