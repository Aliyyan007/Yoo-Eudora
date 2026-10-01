# Auto Bump System

A Discord self-bot system that:
- **Logs into Discord with a real user account** (not a bot account)
- **Joins a server** via invite link
- **Reads messages** in a specified channel
- **Detects when messages are directed at it** using Groq AI (intent detection)
- **Replies with human-like messages** using Groq AI (casual, slang, short, with occasional typos)
- **Auto-bumps the server** using `/bump` slash command via Playwright browser automation (every 2 hours)

## ⚠️ Warning

Discord's Terms of Service prohibit automating user accounts (self-bots). This can result in account bans. **Use an alt/burner account**, not your main. Only use on servers you own or have permission to automate.

## Stack

| Component | Purpose |
|---|---|
| `discord.py-self` | Self-bot library — Discord gateway connection with user token |
| `playwright` | Browser automation for `/bump` slash command |
| `groq` | AI intent detection + human-like reply generation |
| `python-dotenv` | Load secrets from `.env` |
| `pyyaml` | Config file parsing |
| `apscheduler` | (Available, but we use asyncio scheduling) |
| `loguru` | Structured logging |
| `tenacity` | Retry logic for API calls |
| `pydantic` | Config validation |

## Setup

### 1. Install Python 3.12+

Download from [python.org](https://www.python.org/downloads/). During installation, check "Add Python to PATH".

### 2. Install dependencies

```bash
pip install -r requirements.txt
playwright install chromium
```

### 3. Get credentials

#### Discord User Token
1. Open Discord in your web browser (Chrome/Edge)
2. Press F12 to open DevTools
3. Go to Network tab
4. Click any request in the list
5. Find the `authorization` header in the request headers
6. Copy the value — this is your user token

#### Discord Email/Password
The email and password you use to log into Discord. These are used by Playwright to log into the Discord web client for the `/bump` command.

#### Groq API Key
1. Go to [console.groq.com](https://console.groq.com)
2. Create an account
3. Generate an API key

### 4. Configure

Copy `config/.env.example` to `config/.env` and fill in:
```
DISCORD_TOKEN=your_user_token
DISCORD_EMAIL=your_email
DISCORD_PASSWORD=your_password
GROQ_API_KEY=your_groq_key
```

Edit `config/config.yaml`:
- `server_invite`: Your server invite link
- `target_channel_id`: Channel ID where the bot reads and replies (right-click channel → Copy ID, requires Developer Mode in Discord settings)
- `bump_channel_id`: Channel where `/bump` runs (usually a #bump-bot or #commands channel)
- `persona`: Customize the bot's personality, name, style, interests
- `triggers`: Control when the bot replies (mentions, replies, AI intent detection)
- `bump`: Enable/disable, set interval and jitter

### 5. Enable Discord Developer Mode
Settings → Advanced → Developer Mode (toggle on). This lets you right-click channels and messages to copy IDs.

## Running

```bash
python src/main.py
```

## How it works

### Message reading & AI replies
1. The self-bot connects to Discord via gateway using your user token
2. It listens for messages in the target channel
3. For each message, it checks:
   - Was the bot @mentioned or named? → reply
   - Is it a reply to one of the bot's messages? → reply
   - **AI intent detection**: Groq analyzes the conversation context and decides if the message is directed at the bot → reply if confidence > 60%
4. If triggered, it sometimes stays silent (15% chance) or skips (based on reply_probability) to seem more human
5. When replying:
   - Shows typing indicator for a human-like duration (scales with reply length)
   - Generates a short, casual reply using Groq with the persona's style
   - Occasionally adds a typo (8% chance)
   - Sends the reply referencing the original message

### Auto-bump
1. A Playwright browser launches Chromium (visible, not headless — Discord detects headless)
2. Logs into Discord web with email/password
3. Navigates to the bump channel
4. Types `/bump` in the message box, selects from autocomplete, presses Enter
5. Repeats every 2 hours ± 5 minutes jitter
6. Browser session persists in `data/playwright_profile/` so re-logins are rare

### Intent detection
The intent model (`llama-3.1-8b-instant` — fast and cheap) receives:
- The last 15 messages with timestamps and authors
- The new message to evaluate

It returns JSON: `{"directed_at_bot": true/false, "confidence": 0.0-1.0, "reason": "..."}`

This handles slang like "hru", "wbu", "wym", and conversation flow analysis — not just @mentions.

### Reply generation
The reply model (`llama-3.3-70b-versatile` — smarter) receives:
- The conversation context
- The message to reply to
- A detailed system prompt with the persona's style, slang usage, and rules

It generates short, casual, human-like messages with internet slang, lowercase, minimal punctuation, and occasional typos.

## Project structure

```
auto-bump-system/
├── config/
│   ├── config.yaml          # Main configuration
│   └── .env.example         # Template for secrets
├── src/
│   ├── main.py              # Entry point
│   ├── discord_client.py    # Self-bot: message events, replies
│   ├── bump_scheduler.py    # Playwright: /bump on schedule
│   ├── persona.py           # Persona config loader
│   ├── context.py           # Message history formatting
│   ├── ai/
│   │   ├── intent.py        # Intent detection (Groq)
│   │   ├── reply.py         # Reply generation (Groq)
│   │   └── prompts.py       # System prompts
│   └── utils/
│       ├── delays.py        # Human-like timing
│       └── logger.py        # Loguru setup
├── data/
│   └── playwright_profile/  # Browser session (auto-created)
├── logs/
│   └── bot.log              # Log file (auto-created)
├── requirements.txt
└── README.md
```

## Troubleshooting

- **"Target channel not found"**: Make sure the account has joined the server and the channel ID is correct. Enable Developer Mode to copy IDs.
- **Bump fails**: Check that email/password are correct. If Discord asks for 2FA/captcha, complete it manually in the browser window (you have 60 seconds).
- **No replies**: Check that the Groq API key is valid and the models are available. Check logs for intent detection results.
- **Account banned**: This is a self-bot. Use an alt account. Reduce reply frequency. Don't reply to every single message.
