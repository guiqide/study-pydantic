from pydantic import BaseModel
from pydantic_ai import Agent, ToolOutput
from use_model import MyAgent


class Fruit(BaseModel):
  name: str
  color: str

class Vehicle(BaseModel):
  name: str
  wheels: str

agent = MyAgent(
  'deepseek:deepseek-chat',
  output_type=[
    ToolOutput(Fruit, name='return_fruit'),
    ToolOutput(Vehicle, name='return_vehicle'),
  ],
)

result = agent.run_sync('保时捷是什么?')
print(repr(result.output))
