import asyncio, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv("config/.env")
import discord

tok = os.getenv("DISCORD_TOKEN_ROWAN", "").strip()
assert tok, "DISCORD_TOKEN_ROWAN missing"

async def main():
    c = discord.Client()
    @c.event
    async def on_ready():
        print(f"ROWAN OK — logged in as {c.user} (id={c.user.id})")
        await c.close()
    try:
        await asyncio.wait_for(c.start(tok), timeout=90)
    except asyncio.TimeoutError:
        print("ROWAN connect timed out at 90s")
    except Exception as e:
        print(f"ROWAN FAILED: {type(e).__name__}: {e}")

asyncio.run(main())
