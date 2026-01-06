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
from pydantic_ai import Agent, RunContext
from pydantic_ai.providers.deepseek import DeepSeekProvider
from use_model import MyAgent
from what_city import getCity


@dataclass
class Deps:
    client: AsyncClient


model = MyAgent("deepseek:deepseek-chat")

weather_agent = Agent(
    "deepseek:deepseek-chat",
    instructions="你是天气助手：根据用户提供的地点，给出天气预报（必要时调用工具）。",
    deps_type=Deps,
    retries=2,
)


class LatLng(BaseModel):
    lat: float
    lon: float


@weather_agent.tool
async def get_lat_lng(ctx: RunContext[Deps], location_description: str) -> LatLng:
    r = await ctx.deps.client.get(
        "https://geocode.maps.co/search",
        params={"q": location_description, "api_key": os.getenv("GEOCODE_API_KEY")},
    )

    r.raise_for_status()

    res = LatLng.model_validate_json(r.content)
    return res


@weather_agent.tool
async def get_weather(ctx: RunContext[Deps], lat: float, lng: float) -> dict[str, Any]:
    temp_response = await ctx.deps.client.get(
        "https://api.tomorrow.io/v4/weather/forecast?location=%s,%s&apikey=%s" % (lat, lng, os.getenv("TOMORROW_API_KEY")),
        params={
            "location": f"{lat},{lng}",
            "apikey": os.getenv("TOMORROW_API_KEY"),
        },
    )
    temp_response.raise_for_status()
    return {
        "temp": temp_response.json()["timelines"]["daily"][0]["values"]["visibilityAvg"],
    }


async def main():
    async with AsyncClient() as client:
        logfire.instrument_httpx(client, capture_all=True)
        deps = Deps(client=client)
        weatherResult = await weather_agent.run("深圳", deps=deps)
        print("weatherResult.data", weatherResult)
        print(weatherResult.output)
        # latLng = await get_lat_lng('青色的城')
        # print(latLng)


if __name__ == "__main__":
    asyncio.run(main())
