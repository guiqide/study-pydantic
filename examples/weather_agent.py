# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations as _annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from httpx import AsyncClient
from pydantic import BaseModel
from use_model import MyAgent


@dataclass
class Deps:
  client: AsyncClient

weather_agent = MyAgent(Deps)

class LatLng(BaseModel):
  lat: float
  lng: float

async def main():
  async with AsyncClient() as client:

if __name__ == "__main__":
  result = weather_agent.run_sync('深圳天气')
  print(result.output)
  print(result.usage())