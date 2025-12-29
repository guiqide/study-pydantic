
import os

import logfire
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()


model = OpenAIChatModel(
  'deepseek-chat',
  provider=DeepSeekProvider(api_key=os.getenv('DEEPSEEK_API_KEY')),
)

def MyAgent(c):
  return Agent(model=model, output_type=c)