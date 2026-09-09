"""RAG 检索对照实验

三个变体在 36 条标注查询上的命中率对比：
  1. naive-v4   ：text-embedding-v4(1024d) + 原始查询直接检索（现有生产链路）
  2. rewrite-v4 ：text-embedding-v4 + LLM 查询改写后检索
  3. naive-v2   ：text-embedding-v2(1536d，中文优化旧版) + 原始查询

独立使用 volumes/eval_rag.db（不占用主库文件锁），文档切分复用项目的 DocumentSplitterService。

用法：
    NO_PROXY=localhost,127.0.0.1 .venv/bin/python evals/run_rag_eval.py
"""

import asyncio
import json
import sys
import time
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DB_PATH = str(ROOT / "volumes" / "eval_rag.db")
QUERIES_FILE = ROOT / "evals" / "rag_queries.json"
DOCS_DIR = ROOT / "aiops-docs"
RESULTS = ROOT / "evals" / "results" / "rag_eval.json"

TOP_K = 3


class DashEmbed:
    """DashScope embedding（OpenAI 兼容端点），v2 等旧模型不支持 dimensions 参数"""

    def __init__(self, api_key: str, model: str, dimensions: int | None = None):
        self.client = OpenAI(api_key=api_key,
                             base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model = model
        self.dimensions = dimensions

    def _embed(self, texts):
        kwargs = {"model": self.model, "input": texts, "encoding_format": "float"}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        for attempt in range(3):
            try:
                resp = self.client.embeddings.create(**kwargs)
                return [d.embedding for d in resp.data]
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_query(self, text):
        return self._embed([text])[0]


def build_index(embed: DashEmbed, collection: str):
    """用项目同款分块逻辑把 5 篇文档写入独立 Lite 库"""
    from langchain_milvus import Milvus
    from app.services.document_splitter_service import document_splitter_service

    store = Milvus(
        embedding_function=embed,
        collection_name=collection,
        connection_args={"uri": DB_PATH},
        auto_id=True,
        drop_old=True,  # 每次重建，保证对照实验基线一致
        text_field="content",
        vector_field="vector",
        metadata_field="metadata",
    )
    total = 0
    for md in sorted(DOCS_DIR.glob("*.md")):
        docs = document_splitter_service.split_markdown(md.read_text(encoding="utf-8"),
                                                        file_path=str(md))
        for d in docs:
            d.metadata["_source"] = md.name
        store.add_documents(docs)
        total += len(docs)
        print(f"  索引 {md.name}: {len(docs)} chunks")
    print(f"  集合 {collection} 共 {total} chunks")
    return store


async def rewrite_query(llm, query: str) -> str:
    resp = await llm.ainvoke(
        "把下面的用户问题改写为适合在运维知识库中做向量检索的查询语句。"
        "要求：保留关键术语和指标名，去掉口语化表达，只输出改写后的查询，不要解释。\n\n"
        f"用户问题：{query}")
    return resp.content.strip()


async def run_variant(name: str, embed: DashEmbed, use_rewrite: bool) -> dict:
    print(f"\n=== 变体 {name} ===")
    store = build_index(embed, f"eval_{name}")
    queries = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))

    llm = None
    if use_rewrite:
        from langchain_qwq import ChatQwen
        from app.config import config
        llm = ChatQwen(model=config.rag_model, api_key=config.dashscope_api_key, temperature=0)

    hit1 = hit3 = 0
    details = []
    for i, item in enumerate(queries, 1):
        q = item["q"]
        if use_rewrite:
            q_used = await rewrite_query(llm, q)
        else:
            q_used = q
        docs = store.similarity_search(q_used, k=TOP_K)
        got = [d.metadata.get("_source", "") for d in docs]
        h1 = bool(got) and got[0] == item["doc"]
        h3 = item["doc"] in got
        hit1 += h1
        hit3 += h3
        details.append({"q": q, "q_used": q_used, "expect": item["doc"],
                        "got": got, "hit1": h1, "hit3": h3})
        print(f"  [{i}/{len(queries)}] {'✓' if h1 else ('~' if h3 else '✗')} {q[:30]}")
        await asyncio.sleep(0.3)

    n = len(queries)
    return {"variant": name, "n": n,
            "hit@1": round(hit1 / n, 4), "hit@3": round(hit3 / n, 4),
            "details": details}


async def main():
    from app.config import config
    assert config.dashscope_api_key, "需要 DASHSCOPE_API_KEY"

    results = []
    results.append(await run_variant(
        "naive_v4", DashEmbed(config.dashscope_api_key, "text-embedding-v4", 1024), False))
    results.append(await run_variant(
        "rewrite_v4", DashEmbed(config.dashscope_api_key, "text-embedding-v4", 1024), True))
    results.append(await run_variant(
        "naive_v2", DashEmbed(config.dashscope_api_key, "text-embedding-v2", None), False))

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 50)
    print(f"{'变体':<14} {'hit@1':>8} {'hit@3':>8}")
    for r in results:
        print(f"{r['variant']:<14} {r['hit@1']:>8.1%} {r['hit@3']:>8.1%}")
    print(f"\n明细已写入 {RESULTS}")


if __name__ == "__main__":
    asyncio.run(main())
