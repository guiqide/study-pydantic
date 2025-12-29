# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations as _annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import logfire
from httpx import AsyncClient
from pydantic import BaseModel
from pydantic_ai import RunContext
from use_model import MyAgent
from what_city import getCity


@dataclass
class Deps:
  client: AsyncClient

weather_agent = MyAgent('deepseek:deepseek-chat', Deps)

class LatLng(BaseModel):
  lat: float
  lon: float

@weather_agent.tool
async def get_lat_lng(ctx: RunContext[Deps], location_description: str) -> LatLng:
  r = await ctx.deps.client.get(
    'https://geocode.maps.co/reverse?q=%s&api_key=%s' % (location_description, os.getenv('DEEPSEEK_API_KEY')),
    params={'location': location_description},
  )

  r.raise_for_status()

  res = LatLng.model_validate_json(r.content())
  print(res)
  return res

@weather_agent.tool
async def get_weather(ctx: RunContext[Deps], lat: float, lng: float) -> dict[str, Any]:
  temp_response, descr_response = await asyncio.gather(
    ctx.deps.client.get(
      'https://api.tomorrow.io/v4/weather/forecast?location=%s,%s&apikey=%s' % (lat, lng, os.getenv('TOMORROW_API_KEY')),
    )
  )
  temp_response.raise_for_status()
  descr_response.raise_for_status()
  return {
    'temp': temp_response.json()['data']['timelines'][0]['intervals'][0]['values']['temperature'],
    'descr': descr_response.json()['data']['timelines'][0]['intervals'][0]['values']['weatherCode'],
  }

async def main():
  async with AsyncClient() as client:
    logfire.instrument_httpx(client, capture_all=True)
    cityResult = await getCity('青色的城')
    print(cityResult)
    # {lat, lon} = get_lat_lng(ctx, f'{city}, {country}')


if __name__ == "__main__":
  asyncio.run(main())