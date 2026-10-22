"""Telegram order-intake agent demo: tool calling over a catalog + CRM, RAG over company docs."""
from pathlib import Path

from .core import Agent, openai_model
from .crm import Catalog, JsonCRM
from .knowledge import KnowledgeBase

DATA = Path(__file__).resolve().parent.parent / "data"


def build_agent(model, orders_path: str | Path | None = None, log=lambda s: None) -> Agent:
    catalog = Catalog.from_csv(DATA / "catalog.csv")
    return Agent(
        model=model,
        catalog=catalog,
        crm=JsonCRM(orders_path or DATA / "orders.json", catalog),
        kb=KnowledgeBase.from_folder(DATA / "docs"),
        log=log,
    )
