# Yoo Eudora

**Three AI personas that actually live in your Discord server** — Eudora, Isla, and Rowan. Not a bot that answers commands. People who hang out: they chat, remember you, join voice calls, bump the server, and get things done when asked.

> ⚠️ **This project automates Discord user accounts (self-bots), which violates Discord's Terms of Service.** Your account(s) can be banned. The author takes **no responsibility** for anything that happens. **Set up and run entirely at your own risk** — use accounts you're prepared to lose. See [LICENSE](LICENSE) (non-commercial use only).

---

## The Personas

| | | |
|---|---|---|
| **Eudora Edward** | 22, UK — art student. Dry British humor, lowercase texting, french phrases slip in | `DISCORD_TOKEN` |
| **Isla Bennett** | 20 — social, curious, expressive. Photography, music, fashion, cafés, travel | `DISCORD_TOKEN_ISLA` |
| **Rowan Hayes** | 21 — relaxed, witty, slightly sarcastic. Gaming, tech, guitar. Concise and deadpan | `DISCORD_TOKEN_ROWAN` |

One persona is active at a time — the **rotation supervisor** switches accounts every ~2.5h (± jitter), each with its own token, memory namespace, voice, and style. To your server they look like three different people.

## What Makes It Different

- 🎙️ **Voice channel, for real** — full-duplex voice conversation: VAD → streaming ASR → LLM → Fish Audio TTS. Barge-in (interrupt it mid-sentence), mid-utterance backchannels ("mhm/yeah" while you talk), instant cached acks, stale-reply supersede when you keep talking, side-talk detection (won't butt into conversations that aren't for it), per-persona voices and filler sounds, and it calls people by their actual name.
- ⚡ **Action performer** — an isolated worker detects real Discord action requests ("ping John", "react to that", "check their profile") and executes them silently with human-like delay — no "done!" spam.
- 🧠 **Real memory** — per-user facts, hobbies, relationships, memorable chats; contradiction-aware updates; daily stale-data sweep so the DB never bloats. Persists to Cloudflare D1 with local JSON fallback.
- 💬 **Wise engagement** — dead-chat revival, re-engagement pings with hard safety gates (cooldowns, daily caps, unanswered-streak muting), proactive conversation, welcome messages, sticker/GIF reactions.
- 🔄 **Auto-bump** — per-bot cooldown-aware scheduler driving `/bump` across Disboard, Bumper, Carl Bot, OneBump, Bump4You — state persisted across restarts and rotation.
- 🛡️ **Human-safe** — no IRL meetups, no DM promises, no flirty escalation, owner-exempt abuse handling, question dodging respected.

## Setup

> **Don't want to self-host?** Skip the pain — **talk to Aliyyan** and he'll set it up for you: [Discord server](https://discord.gg/FvVWf4a7TY) · [aliyyan.com](https://aliyyan.com)

### Requirements
- Python 3.12+
- Discord user token(s) — one per persona account you want to run
- [Groq](https://console.groq.com) API key(s) — free tier works; more keys = more capacity (10 recommended for 24/7)
- [Fish Audio](https://fish.audio) API key — for voice (optional; text-only works without it)

### Quick start
```bash
pip install -r requirements.txt
copy config\.env.example config\.env    # then fill in your tokens + keys
python src\main.py
```

`config/.env.example` is fully documented — persona voice IDs are already prefilled (they're public voice models). Only secrets (tokens, API keys, D1 creds) need your own values.

### Deploy to Render
`render.yaml` + `Dockerfile` included — point a Render service at the repo, add your env vars, done. A keep-alive HTTP server is built in for free-tier hosting.

## Find Me / Get Help

**Setup requests, questions, or just want to see it live — talk to Aliyyan:**

- 🌐 Portfolio: [aliyyan.com](https://aliyyan.com)
- 💬 Discord: [discord.gg/FvVWf4a7TY](https://discord.gg/FvVWf4a7TY)
- 🐙 GitHub: [@Aliyyan007](https://github.com/Aliyyan007)
- 📸 Instagram: [@aliyyan007](https://instagram.com/aliyyan007)
- 🎮 Steam: [aliyyan007](https://steamcommunity.com/id/aliyyan007)
- 🐦 X: [@aliyyan007](https://x.com/aliyyan007)
- 🎧 Spotify: [Aliyyan](https://open.spotify.com/user/31gjqqqiavkbdsnthiuuim6f734a)

## License

**Non-commercial, source-available** — free to use, study, and modify for personal/educational purposes with credit. **No commercial use without written permission** — unauthorized monetization may result in legal action. See [LICENSE](LICENSE).

---

*Developer: **Aliyyan** — [aliyyan.com](https://aliyyan.com)*
