"""The agent loop: an OpenAI-compatible chat model decides which tools to call.

Works with OpenAI, DeepSeek, OpenRouter, a local Ollama/vLLM server — anything that
speaks the /chat/completions tool-calling protocol (set OPENAI_BASE_URL and MODEL).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .crm import Catalog, JsonCRM
from .knowledge import KnowledgeBase

SYSTEM_PROMPT = """Ты — менеджер по приёму заявок оптового поставщика для баров и ресторанов.
Правила:
1. Товары ищи только через search_catalog. Никогда не придумывай SKU, цены и остатки.
2. Если позиция неоднозначна (несколько похожих товаров) — коротко уточни у клиента, какую именно.
3. Перед create_order проверь: все позиции найдены, есть количество и дата доставки. Чего-то нет — спроси.
4. Вопросы про доставку, оплату, возвраты, минимальный заказ — отвечай только по search_docs и указывай источник в скобках.
   Если в документах ответа нет — честно скажи и предложи связаться с менеджером.
5. Отвечай кратко, по-русски, без markdown-таблиц."""

TOOLS = [
    {"type": "function", "function": {
        "name": "search_catalog",
        "description": "Найти товары в каталоге по названию в свободной форме (допускаются опечатки и сокращения).",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "create_order",
        "description": "Создать заявку в CRM. Вызывать только когда известны SKU, количества и дата доставки.",
        "parameters": {"type": "object", "properties": {
            "items": {"type": "array", "items": {"type": "object", "properties": {
                "sku": {"type": "string"}, "qty": {"type": "number"}}, "required": ["sku", "qty"]}},
            "delivery_date": {"type": "string", "description": "YYYY-MM-DD или как сказал клиент"},
            "comment": {"type": "string"}},
            "required": ["items", "delivery_date"]}}},
    {"type": "function", "function": {
        "name": "search_docs",
        "description": "Поиск по базе знаний компании: доставка, оплата, возвраты, условия.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]


class ChatModel(Protocol):
    def __call__(self, messages: list[dict], tools: list[dict]) -> dict:
        """Return an assistant message dict: {"content": str|None, "tool_calls": [...]|None}."""


def openai_model(model: str, client: Any | None = None) -> ChatModel:
    from openai import OpenAI  # imported lazily so tests run without network or keys

    client = client or OpenAI()

    def call(messages: list[dict], tools: list[dict]) -> dict:
        r = client.chat.completions.create(model=model, messages=messages, tools=tools, temperature=0.2)
        m = r.choices[0].message
        calls = [{"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                 for c in (m.tool_calls or [])]
        return {"content": m.content, "tool_calls": calls or None}

    return call


@dataclass
class Agent:
    model: ChatModel
    catalog: Catalog
    crm: JsonCRM
    kb: KnowledgeBase
    max_steps: int = 6
    histories: dict[str, list[dict]] = field(default_factory=dict)
    log: Callable[[str], None] = lambda s: None

    def _tool(self, name: str, args: dict, customer: str) -> Any:
        if name == "search_catalog":
            return self.catalog.search(args["query"])
        if name == "create_order":
            return self.crm.create_order(customer, args["items"], args["delivery_date"], args.get("comment", ""))
        if name == "search_docs":
            return [{"source": f"{c.source} › {c.heading}", "text": c.text} for _, c in self.kb.search(args["query"])]
        return {"error": f"unknown tool {name}"}

    def reply(self, customer: str, text: str) -> str:
        history = self.histories.setdefault(customer, [{"role": "system", "content": SYSTEM_PROMPT}])
        history.append({"role": "user", "content": text})
        for _ in range(self.max_steps):
            msg = self.model(history, TOOLS)
            history.append({"role": "assistant", "content": msg.get("content"), **({"tool_calls": msg["tool_calls"]} if msg.get("tool_calls") else {})})
            if not msg.get("tool_calls"):
                return msg.get("content") or ""
            for call in msg["tool_calls"]:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"]["arguments"] or "{}")
                    result = self._tool(name, args, customer)
                except Exception as e:  # a bad tool call must not kill the conversation
                    result = {"error": str(e)}
                self.log(f"[tool] {name}({call['function']['arguments']}) -> {json.dumps(result, ensure_ascii=False)[:300]}")
                history.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False)})
        return "Не удалось обработать заявку автоматически — передаю менеджеру."
