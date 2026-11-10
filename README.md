# AI-агент приёма заявок для Telegram

Демо-проект: Telegram-бот на LLM, который принимает заявки от клиентов оптового поставщика в свободной форме — текстом или голосом — и отвечает на вопросы по базе знаний компании.

Пример запроса: «2 ящика швепса и 3 кило лаймов на пятницу».

Что делает агент:
1. Находит товары в каталоге через `search_catalog`. Опечатки, сокращения и синонимы («швепс», «кола зеро») не мешают поиску.
2. Если позиция неоднозначна (например, «швепс» — это 0,33 или 1 л?), уточняет у клиента.
3. Создаёт заявку в CRM через `create_order`. Проверяет каждый SKU и остатки на складе. Выдуманный SKU отклоняет. Нехватку товара помечает предупреждением, а не скрывает.
4. Отвечает про доставку, оплату и возвраты только по документам компании (`search_docs`, RAG на BM25) и указывает источник.
5. Пишет менеджеру в Telegram о каждой новой заявке.


## Local portfolio demo

I run this project locally on my computer with my own environment configuration. It can be demonstrated in Telegram while the process is active; the repository contains setup instructions and reproducible demo data. The CSV catalogue and JSON CRM are demonstration integrations. Keep credentials in the local environment file.

## Архитектура
```
Telegram (aiogram 3) ── текст / голос (Whisper) ──► Agent.reply()
                                                    │  цикл tool calling (до 6 шагов)
                                   OpenAI-совместимая LLM
                     ┌───────────────┼────────────────┐
             search_catalog     create_order       search_docs
             (каталог CSV,      (JsonCRM →         (BM25 по Markdown,
              нечёткий поиск)    orders.json)        цитата источника)
```
- `agent/core.py` — системный промпт, описание инструментов, цикл агента. Ошибки в вызове инструмента не роняют диалог. Если шагов слишком много, заявка уходит менеджеру.
- `agent/crm.py` — каталог и хранение заявок. Чтобы подключить amoCRM, Bitrix24, Google Таблицы или API клиента, достаточно заменить `JsonCRM`, реализовав тот же метод `create_order`.
- `agent/knowledge.py` — RAG без внешних сервисов. Векторную базу можно подключить вместо `search`.
- `bot.py` — Telegram-интерфейс; `cli.py` — чат в терминале.

Модель любая с OpenAI-совместимым API и поддержкой tool calling: OpenAI, DeepSeek, OpenRouter, локальная Ollama или vLLM. Нужно задать `OPENAI_BASE_URL` и `MODEL`.

## Запуск
```bash
pip install -r requirements.txt
cp .env.example .env        # BOT_TOKEN, OPENAI_API_KEY, MODEL, MANAGER_CHAT_ID
python cli.py               # проверить в терминале
python bot.py               # запустить Telegram-бота
```

## Тесты
```bash
python -m unittest -v
```
Девять офлайн-тестов. Вместо LLM в них работает скриптовая модель, поэтому сеть и ключи не нужны. Проверяются:
- цикл вызова инструментов;
- нечёткий поиск по каталогу;
- что выдуманные SKU отклоняются, а нехватка на складе помечается;
- устойчивость к битым аргументам;
- что RAG находит нужный раздел с источником;
- передача менеджеру при превышении лимита шагов.

> Каталог и условия в `data/` вымышленные, для демонстрации. Качество ответов на живой модели зависит от выбранной LLM и проверяется на данных заказчика.

---
Арсений Атрохов · [GitHub](https://github.com/CondrusArsen) · Telegram [@qintin](https://t.me/qintin) · собрано с ИИ-агентами (Claude Code).
