"""One-shot D1 schema initializer."""
import asyncio
import os
from dotenv import load_dotenv

load_dotenv(os.path.join("config", ".env"))

from src.ai.d1_client import get_d1_client


async def main():
    client = get_d1_client()
    if not await client.is_available():
        print("D1 not available — check CF_D1_* credentials in config/.env")
        return
    await client.init_schema()
    await client.close()
    print("D1 schema initialized successfully")


if __name__ == "__main__":
    asyncio.run(main())
