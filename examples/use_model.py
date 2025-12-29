
import os

import logfire
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()


model = OpenAIChatModel(
  'deepseek-chat',
  provider=DeepSeekProvider(api_key=os.getenv('DEEPSEEK_API_KEY')),
)

def MyAgent(
    model_name: str,
    output_type=str,
    deps_type=None,
    instructions=None,
    retries: int = 1,
):
  """
  统一封装 Agent 创建入口，默认使用 DeepSeek（OpenAI 兼容接口）。

  - model_name: 例如 'deepseek:deepseek-chat'
  - output_type: Agent 输出类型（默认 str / 也可传 BaseModel）
  - deps_type: 依赖类型（例如 @dataclass Deps），用于 tool 的 ctx.deps
  """
  match model_name:
    case 'deepseek:deepseek-chat':
      model = OpenAIChatModel(
        'deepseek-chat',
        provider=DeepSeekProvider(api_key=os.getenv('DEEPSEEK_API_KEY')),
      )
    case _:
      raise ValueError(f'Invalid model name: {model_name}')

  return Agent(
      model=model,
      output_type=output_type,
      deps_type=deps_type,
      instructions=instructions,
      retries=retries,
  )
