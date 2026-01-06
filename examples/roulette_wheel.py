import os

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

model = OpenAIChatModel(
  'deepseek-chat',
  provider=DeepSeekProvider(api_key=os.getenv('DEEPSEEK_API_KEY')),
)

roulette_wheel_agent = Agent(
  model=model,
  deps_type=int,
  output_type=bool,
  system_prompt=('使用`roulette_wheel`功能查看客户是否中奖，基于他们提供的号码。')
)

@roulette_wheel_agent.tool
async def roulette_wheel(ctx: RunContext[int], square: int) -> str:
  return '赢家' if square == ctx.deps else '输家'

def main():
  success_number = 18
  result = roulette_wheel_agent.run_sync('我压在18号格子上', deps=success_number)
  print(result.output)

  result = roulette_wheel_agent.run_sync('我打赌5是赢家', deps=success_number)
  print(result.output)

if __name__ == "__main__":
  main()