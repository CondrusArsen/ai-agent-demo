"""Offline tests: a scripted model stands in for the LLM, so the tool loop, catalog
matching, order validation and RAG are checked without network or API keys.
Run: python -m unittest -v"""
import json
import tempfile
import unittest
from pathlib import Path

from agent import build_agent
from agent.crm import Catalog
from agent.knowledge import KnowledgeBase

DATA = Path(__file__).resolve().parent.parent / "data"


def call(name, **args):
    return {"id": f"c-{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


class Scripted:
    """Returns pre-baked assistant messages and records what the agent fed back."""
    def __init__(self, *steps):
        self.steps, self.seen = list(steps), []

    def __call__(self, messages, tools):
        self.seen.append([m for m in messages if m["role"] == "tool"])
        step = self.steps.pop(0)
        return step(messages) if callable(step) else step


class CatalogTest(unittest.TestCase):
    cat = Catalog.from_csv(DATA / "catalog.csv")

    def test_fuzzy_alias_and_typo(self):
        self.assertEqual(self.cat.search("швепс 0.33")[0]["sku"], "TN-033")
        self.assertEqual(self.cat.search("лаймы")[0]["sku"], "LI-1")
        self.assertEqual(self.cat.search("кока кола зеро")[0]["sku"], "CL-Z033")
        self.assertEqual(self.cat.search("гренадинн")[0]["sku"], "SY-GR")

    def test_unknown_returns_nothing_useful(self):
        self.assertFalse(any(r["sku"].startswith("TN") for r in self.cat.search("водка премиум")))


class KnowledgeTest(unittest.TestCase):
    kb = KnowledgeBase.from_folder(DATA / "docs")

    def test_retrieves_right_section(self):
        self.assertEqual(self.kb.search("какая минимальная сумма для бесплатной доставки")[0][1].heading, "Доставка")
        self.assertEqual(self.kb.search("возврат зелени при порче")[0][1].heading, "Возвраты и бой")
        self.assertEqual(self.kb.search("отсрочка платежа")[0][1].heading, "Оплата")


class AgentLoopTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "orders.json"

    def test_order_flow_calls_tools_and_saves_order(self):
        model = Scripted(
            {"content": None, "tool_calls": [call("search_catalog", query="швепс 0.33"), call("search_catalog", query="лайм")]},
            {"content": None, "tool_calls": [call("create_order", items=[{"sku": "TN-033", "qty": 2}, {"sku": "LI-1", "qty": 3}], delivery_date="2026-10-24")]},
            lambda msgs: {"content": "Заявка " + json.loads(msgs[-1]["content"])["order"]["id"] + " принята", "tool_calls": None},
        )
        agent = build_agent(model, self.tmp)
        answer = agent.reply("Бар «Ромашка»", "2 ящика швепса и 3 кг лаймов на пятницу")
        self.assertEqual(answer, "Заявка Z-0001 принята")
        saved = json.loads(self.tmp.read_text(encoding="utf-8"))
        self.assertEqual(saved[0]["total"], 2 * 1490 + 3 * 390)
        self.assertEqual(saved[0]["customer"], "Бар «Ромашка»")
        # search results really were fed back to the model before it ordered
        catalog_result = json.loads(model.seen[1][0]["content"])
        self.assertEqual(catalog_result[0]["sku"], "TN-033")

    def test_stock_shortage_is_flagged_not_hidden(self):
        agent = build_agent(Scripted(), self.tmp)
        r = agent.crm.create_order("x", [{"sku": "SY-GR", "qty": 50}], "завтра")
        self.assertTrue(r["ok"])
        self.assertIn("на складе 14", r["order"]["warnings"][0])

    def test_invented_sku_is_rejected(self):
        agent = build_agent(Scripted(), self.tmp)
        r = agent.crm.create_order("x", [{"sku": "FAKE-1", "qty": 1}], "завтра")
        self.assertFalse(r["ok"])
        self.assertFalse(self.tmp.exists())

    def test_bad_tool_arguments_do_not_crash(self):
        bad = {"id": "c1", "type": "function", "function": {"name": "create_order", "arguments": "{not json"}}
        model = Scripted({"content": None, "tool_calls": [bad]}, {"content": "Уточните, пожалуйста, заявку", "tool_calls": None})
        agent = build_agent(model, self.tmp)
        self.assertEqual(agent.reply("x", "что-то"), "Уточните, пожалуйста, заявку")
        self.assertIn("error", model.seen[1][0]["content"])

    def test_docs_answer_gets_source(self):
        model = Scripted(
            {"content": None, "tool_calls": [call("search_docs", query="до скольки можно отменить заявку")]},
            lambda msgs: {"content": json.loads(msgs[-1]["content"])[0]["source"], "tool_calls": None},
        )
        agent = build_agent(model, self.tmp)
        self.assertEqual(agent.reply("x", "до скольки можно отменить заявку?"), "conditions.md › Изменение и отмена заявки")

    def test_step_limit_hands_over_to_manager(self):
        loop = {"content": None, "tool_calls": [call("search_catalog", query="кола")]}
        agent = build_agent(Scripted(*[loop] * 10), self.tmp)
        agent.max_steps = 3
        self.assertIn("менеджеру", agent.reply("x", "кола"))


if __name__ == "__main__":
    unittest.main()
