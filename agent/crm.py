"""Catalog lookup and order storage — the "actions" the agent can take.

`JsonCRM` writes orders to a JSON file. A real deployment swaps it for amoCRM, Bitrix24,
Google Sheets or the client's own API by implementing the same two methods.
"""
from __future__ import annotations

import csv
import difflib
import json
import re
import time
from pathlib import Path


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zа-яё0-9 ]", " ", s.lower())).strip()


class Catalog:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    @classmethod
    def from_csv(cls, path: str | Path) -> "Catalog":
        with open(path, encoding="utf-8") as f:
            return cls([dict(r) for r in csv.DictReader(f)])

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Fuzzy match on name + aliases; tolerant to typos and word order."""
        q = _norm(query)
        q_words = set(q.split())
        scored = []
        for row in self.rows:
            hay = _norm(row["name"] + " " + row.get("aliases", ""))
            ratio = difflib.SequenceMatcher(None, q, hay).ratio()
            words = set(hay.split())
            overlap = sum(1 for w in q_words if any(difflib.SequenceMatcher(None, w, h).ratio() > 0.8 for h in words))
            score = ratio + overlap
            if overlap or ratio > 0.5:
                scored.append((score, row))
        scored.sort(key=lambda x: -x[0])
        return [
            {"sku": r["sku"], "name": r["name"], "unit": r["unit"], "price": float(r["price"]), "in_stock": int(r["stock"])}
            for _, r in scored[:limit]
        ]

    def get(self, sku: str) -> dict | None:
        return next((r for r in self.rows if r["sku"] == sku), None)


class JsonCRM:
    def __init__(self, path: str | Path, catalog: Catalog):
        self.path = Path(path)
        self.catalog = catalog

    def _load(self) -> list[dict]:
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []

    def create_order(self, customer: str, items: list[dict], delivery_date: str, comment: str = "") -> dict:
        lines, problems, total = [], [], 0.0
        for it in items:
            row = self.catalog.get(it.get("sku", ""))
            qty = float(it.get("qty", 0))
            if not row:
                problems.append(f"нет SKU {it.get('sku')}")
                continue
            if qty <= 0:
                problems.append(f"{row['name']}: количество должно быть больше 0")
                continue
            if qty > int(row["stock"]):
                problems.append(f"{row['name']}: на складе {row['stock']} {row['unit']}, запрошено {qty:g}")
            price = float(row["price"])
            lines.append({"sku": row["sku"], "name": row["name"], "qty": qty, "unit": row["unit"], "sum": round(qty * price, 2)})
            total += qty * price
        if not lines:
            return {"ok": False, "problems": problems or ["пустой заказ"]}
        orders = self._load()
        order = {
            "id": f"Z-{len(orders) + 1:04d}",
            "created": time.strftime("%Y-%m-%d %H:%M"),
            "customer": customer,
            "delivery_date": delivery_date,
            "items": lines,
            "total": round(total, 2),
            "comment": comment,
            "warnings": problems,
            "status": "new",
        }
        orders.append(order)
        self.path.write_text(json.dumps(orders, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"ok": True, "order": order}
