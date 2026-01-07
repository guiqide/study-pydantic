"""
RAG 示例（PydanticAI）：用“向量检索”增强聊天 Agent 的回答能力。

启动 pgvector（PostgreSQL + 向量扩展）：

    mkdir postgres-data
    docker run --rm -e POSTGRES_PASSWORD=postgres \
        -p 54320:5432 \
        -v `pwd`/postgres-data:/var/lib/postgresql/data \
        pgvector/pgvector:pg17

构建检索数据库：

    uv run -m pydantic_ai_examples.rag build

向 Agent 提问（RAG 问答）：

    uv run -m pydantic_ai_examples.rag search "如何配置 logfire 与 FastAPI 一起工作？"
"""

from __future__ import annotations as _annotations

import asyncio
import logging
import pdb
import re
import sys
import unicodedata
from contextlib import asynccontextmanager
from dataclasses import dataclass

import asyncpg
import httpx
import logfire
import ollama
import pydantic_core
from httpx import AsyncClient
from pydantic import TypeAdapter
from pydantic_ai import Agent, RunContext
from typing_extensions import AsyncGenerator
from use_model import MyAgent

# 'if-token-present' 表示：如果你没有配置 logfire token，就不会发送任何数据（示例仍可运行）
logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_asyncpg()
logfire.instrument_pydantic_ai()

timeout = httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0)  # LLM 场景建议 ≥ 60s


@dataclass
class Deps:
    client: AsyncClient(timeout=timeout)
    pool: asyncpg.Pool


agent = MyAgent("deepseek:deepseek-chat", deps_type=Deps)


@agent.tool
async def retrieve(context: RunContext[Deps], search_query: str) -> str:
    """根据检索 query 召回相关文档片段（向量检索）。

    Args:
        context: 调用上下文（RunContext），包含 deps（HTTPX client、数据库连接池等）。
        search_query: 检索 query（通常就是用户的问题或其改写）。
    """
    with logfire.span("create embedding for {search_query=}", search_query=search_query):
        embedding = await context.deps.client.post(
            "http://localhost:11434/api/embed",
            json={
                "input": search_query,
                "model": "qwen3-embedding:8b",
            },
        )
        embedding = embedding.json()
    assert len(embedding["embeddings"]) == 1, f"Expected 1 embedding, got {len(embedding['embeddings'])}, doc query: {search_query!r}"
    embedding = embedding["embeddings"][0]
    embedding_json = pydantic_core.to_json(embedding).decode()
    rows = await context.deps.pool.fetch(
        "SELECT url, title, content FROM doc_sections ORDER BY embedding <-> $1 LIMIT 8",
        embedding_json,
    )
    str = "\n\n".join(f'# {row["title"]}\nDocumentation URL:{row["url"]}\n\n{row["content"]}\n' for row in rows)
    return str


async def run_agent(question: str):
    """运行入口：执行一次 RAG 问答。"""
    client = AsyncClient(timeout=timeout)
    logfire.instrument_httpx(client)

    logfire.info('Asking "{question}"', question=question)

    async with database_connect(False) as pool:
        deps = Deps(client=client, pool=pool)
        answer = await agent.run(question, deps=deps)
    print(answer.output)


#######################################################
# 以下内容用于：准备检索数据库，以及一些辅助工具。        #
#######################################################

# 文档切片 JSON 来源：
# `https://gist.github.com/samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992`
DOCS_JSON = (
    "https://gist.githubusercontent.com/"
    "samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992/raw/"
    "80c5925c42f1442c24963aaf5eb1a324d47afe95/logfire_docs.json"
)


async def build_search_db():
    """构建检索数据库（建表、生成 embedding、写入 pgvector）。"""
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(DOCS_JSON)
        response.raise_for_status()
    sections = sessions_ta.validate_json(response.content)

    client = AsyncClient(timeout=timeout)
    logfire.instrument_httpx(client)

    async with database_connect(True) as pool:
        with logfire.span("create schema"):
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(DB_SCHEMA)

        sem = asyncio.Semaphore(10)
        async with asyncio.TaskGroup() as tg:
            for section in sections:
                tg.create_task(insert_doc_section(sem, client, pool, section))


async def insert_doc_section(
    sem: asyncio.Semaphore,
    client: AsyncClient(timeout=timeout),
    pool: asyncpg.Pool,
    section: DocsSection,
) -> None:
    async with sem:
        url = section.url()
        exists = await pool.fetchval("SELECT 1 FROM doc_sections WHERE url = $1", url)
        if exists:
            logfire.info("Skipping {url=}", url=url)
            return

        with logfire.span("create embedding for {url=}", url=url):
            embedding = await client.post(
                "http://localhost:11434/api/embed",
                # headers={"Content-Type": "application/json"},
                json={
                    "input": section.embedding_content(),
                    "model": "qwen3-embedding:8b",
                },
            )
            embedding = embedding.json()
        assert len(embedding["embeddings"]) == 1, f"Expected 1 embedding, got {len(embedding['embeddings'])}, doc section: {section}"
        embedding = embedding["embeddings"][0]
        embedding_json = pydantic_core.to_json(embedding).decode()

        await pool.execute(
            "INSERT INTO doc_sections (url, title, content, embedding) VALUES ($1, $2, $3, $4)",
            url,
            section.title,
            section.content,
            embedding_json,
        )


@dataclass
class DocsSection:
    id: int
    parent: int | None
    path: str
    level: int
    title: str
    content: str

    def url(self) -> str:
        url_path = re.sub(r"\.md$", "", self.path)
        return f'https://logfire.pydantic.dev/docs/{url_path}/#{slugify(self.title, "-")}'

    def embedding_content(self) -> str:
        return "\n\n".join((f"path: {self.path}", f"title: {self.title}", self.content))


logging.basicConfig(level=logging.INFO)
sessions_ta = TypeAdapter(list[DocsSection])


# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
@asynccontextmanager
async def database_connect(
    create_db: bool = False,
) -> AsyncGenerator[asyncpg.Pool, None]:
    server_dsn, database = (
        "postgresql://postgres:o5h^mDALrY^qH53h@9.134.11.191:5432",
        "pydantic_ai_rag",
    )
    if create_db:
        with logfire.span("check and create DB"):
            conn = await asyncpg.connect(server_dsn)
            try:
                db_exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", database)
                if not db_exists:
                    await conn.execute(f"CREATE DATABASE {database}")
            finally:
                await conn.close()

    pool = await asyncpg.create_pool(f"{server_dsn}/{database}")
    try:
        yield pool
    finally:
        await pool.close()


DB_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS doc_sections (
    id serial PRIMARY KEY,
    url text NOT NULL UNIQUE,
    title text NOT NULL,
    content text NOT NULL,
    -- qwen3-embedding:8b returns a vector of 4096 floats
    embedding vector(4096) NOT NULL
);
"""


def slugify(value: str, separator: str, unicode: bool = False) -> str:
    """把字符串转成 URL 友好的 slug。"""
    # Taken unchanged from https://github.com/Python-Markdown/markdown/blob/3.7/markdown/extensions/toc.py#L38
    if not unicode:
        # Replace Extended Latin characters with ASCII, i.e. `žlutý` => `zluty`
        value = unicodedata.normalize("NFKD", value)
        value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(rf"[{separator}\s]+", separator, value)


if __name__ == "__main__":
    logging.info(f"Searching for {sys.argv}")
    action = sys.argv[1] if len(sys.argv) > 1 else None
    if action == "build":
        asyncio.run(build_search_db())
    elif action == "search":
        if len(sys.argv) == 3:
            q = sys.argv[2]
        else:
            q = "How do I configure logfire to work with FastAPI?"
        asyncio.run(run_agent(q))
    else:
        print(
            "uv run --extra examples -m pydantic_ai_examples.rag build|search",
            file=sys.stderr,
        )
        sys.exit(1)
