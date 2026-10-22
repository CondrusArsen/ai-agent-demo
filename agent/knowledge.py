"""Small RAG layer: splits Markdown documents into chunks and ranks them with BM25.

No embeddings service is needed, so the demo runs offline; swapping in a vector store
only means replacing `KnowledgeBase.search`.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

TOKEN = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)
STOP = {"и", "в", "во", "на", "до", "по", "с", "со", "к", "о", "об", "от", "за", "из", "у", "не", "ли", "или",
        "а", "но", "что", "как", "это", "можно", "нужно", "какая", "какой", "какие", "мне", "вы", "мы", "я"}


def tokenize(text: str) -> list[str]:
    # Crude stemming: keep the first 6 letters so «доставка», «доставки», «доставкой» match.
    return [t.lower()[:6] for t in TOKEN.findall(text) if t.lower() not in STOP]


@dataclass
class Chunk:
    source: str      # file name
    heading: str     # nearest Markdown heading
    text: str


class KnowledgeBase:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.docs = [tokenize(c.heading + " " + c.text) for c in chunks]
        self.avgdl = sum(map(len, self.docs)) / max(1, len(self.docs))
        df: dict[str, int] = {}
        for d in self.docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    @classmethod
    def from_folder(cls, folder: str | Path) -> "KnowledgeBase":
        chunks: list[Chunk] = []
        for path in sorted(Path(folder).glob("*.md")):
            heading, buf = path.stem, []
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("#"):
                    if buf:
                        chunks.append(Chunk(path.name, heading, "\n".join(buf).strip()))
                        buf = []
                    heading = line.lstrip("#").strip()
                elif line.strip():
                    buf.append(line)
            if buf:
                chunks.append(Chunk(path.name, heading, "\n".join(buf).strip()))
        return cls(chunks)

    def search(self, query: str, k: int = 3, k1: float = 1.5, b: float = 0.75) -> list[tuple[float, Chunk]]:
        q = tokenize(query)
        scored = []
        for doc, chunk in zip(self.docs, self.chunks):
            score = 0.0
            for t in q:
                tf = doc.count(t)
                if tf:
                    score += self.idf.get(t, 0) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / self.avgdl))
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]
