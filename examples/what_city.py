# -*- coding: utf-8 -*-
"""
最小 PydanticAI 示例（无需 API Key）。

运行：
  source .venv/bin/activate
  python examples/minimal_pydantic_ai.py
"""

from pydantic import BaseModel
from use_model import MyAgent


class MyModel(BaseModel):
  city: str
  country: str

agent = MyAgent('deepseek:deepseek-chat', output_type=MyModel)

def getCity(city: str = '青色的城') -> MyModel:
  result = agent.run_sync(city)
  print(result.output)
  print(result.usage())
  return result.output

if __name__ == "__main__":
  getCity()