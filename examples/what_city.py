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

agent = MyAgent(MyModel)

if __name__ == "__main__":
  result = agent.run_sync('青色的城')
  print(result.output)
  print(result.usage())

