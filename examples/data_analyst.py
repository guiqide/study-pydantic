from dataclasses import dataclass, field

import datasets
import duckdb
import pandas as pd
from pydantic_ai import Agent, ModelRetry, RunContext
from use_model import MyAgent


@dataclass
class AnalystAgentDeps:
    output: dict[str, pd.DataFrame] = field(default_factory=dict)

    def store(self, value: pd.DataFrame) -> str:
        """把输出 DataFrame 存到 deps 里，并返回一个引用名（如 Out[1]），供大模型在后续对话中引用。"""
        ref = f"Out[{len(self.output) + 1}]"
        self.output[ref] = value
        return ref

    def get(self, ref: str) -> pd.DataFrame:
        if ref not in self.output:
            raise ModelRetry(f"错误：{ref} 不是有效的变量引用。请检查前面的消息并重试。")
        return self.output[ref]


analyst_agent = MyAgent(
    "deepseek:deepseek-chat",
    deps_type=AnalystAgentDeps,
    instructions="你是一名数据分析师，你的工作是根据用户需求对数据进行分析，并给出结论。返回中文语言的回答。",
)


@analyst_agent.tool
def load_dataset(
    ctx: RunContext[AnalystAgentDeps],
    path: str,
    split: str = "train",
) -> str:
    """从 HuggingFace 加载指定数据集的某个 split。

    Args:
        ctx: PydanticAI 的运行上下文（RunContext）
        path: 数据集名称，格式为 `<user_name>/<dataset_name>`
        split: 要加载的数据集 split（默认："train"）
    """
    # 开始：从 HuggingFace 加载数据
    builder = datasets.load_dataset_builder(path)  # pyright: ignore[reportUnknownMemberType]
    splits: dict[str, datasets.SplitInfo] = builder.info.splits or {}  # pyright: ignore[reportUnknownMemberType]
    if split not in splits:
        raise ModelRetry(f'{split} is not valid for dataset {path}. Valid splits are {",".join(splits.keys())}')

    builder.download_and_prepare()  # pyright: ignore[reportUnknownMemberType]
    dataset = builder.as_dataset(split=split)
    assert isinstance(dataset, datasets.Dataset)
    dataframe = dataset.to_pandas()
    assert isinstance(dataframe, pd.DataFrame)
    # 结束：从 HuggingFace 加载数据

    # 将 dataframe 存入 deps，并拿到一个类似 "Out[1]" 的引用名
    ref = ctx.deps.store(dataframe)
    # 构造加载结果摘要，返回给模型
    output = [
        f"已加载数据集为 `{ref}`。",
        f"描述：{dataset.info.description}" if dataset.info.description else None,
        f"特征：{dataset.info.features!r}" if dataset.info.features else None,
    ]
    return "\n".join(filter(None, output))


@analyst_agent.tool
def run_duckdb(ctx: RunContext[AnalystAgentDeps], dataset: str, sql: str) -> str:
    """在 DataFrame 上运行 DuckDB SQL 查询。

    注意：DuckDB SQL 中使用的虚拟表名必须是 `dataset`。

    Args:
        ctx: PydanticAI 的运行上下文（RunContext）
        dataset: DataFrame 的引用名（如 Out[1]）
        sql: 要执行的 DuckDB SQL
    """
    data = ctx.deps.get(dataset)
    result = duckdb.query_df(df=data, virtual_table_name="dataset", sql_query=sql)
    # 将结果也用引用名返回（因为 SQL 可能选出很多行，直接返回可能很大）
    ref = ctx.deps.store(result.df())  # pyright: ignore[reportUnknownMemberType]
    return f"已执行 SQL，结果为 `{ref}`"


@analyst_agent.tool
def display(ctx: RunContext[AnalystAgentDeps], name: str) -> str:
    """最多展示该 DataFrame 的前 5 行。"""
    dataset = ctx.deps.get(name)
    return dataset.head().to_string()  # pyright: ignore[reportUnknownMemberType]


if __name__ == "__main__":
    deps = AnalystAgentDeps()
    result = analyst_agent.run_sync(
        user_prompt="统计数据集 `cornell-movie-review-data/rotten_tomatoes` 里有多少条负面评论。并展示前5条负面评论。",
        deps=deps,
    )
    print(result.output)
