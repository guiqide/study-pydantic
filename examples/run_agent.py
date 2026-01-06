# 流式输出示例
import asyncio

from use_model import MyAgent

agent = MyAgent("deepseek:deepseek-chat")

result_sync = agent.run_sync("意大利的首都是哪里?")
print(result_sync.output)


async def main():
    async with agent.run_stream("英国的首都是哪里?") as response:
        async for text in response.stream_text():
            print(text, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
