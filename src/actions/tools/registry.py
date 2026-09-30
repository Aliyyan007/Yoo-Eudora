"""Tool registry: Groq function-calling schemas + dispatch.

Two schema sets:
  - FULL: 55 tools with detailed descriptions (command/event mode).
  - CHAT: 39 tools with minimal descriptions (chat mode, saves tokens).

The agent calls `dispatch(ctx, name, args)` to run a tool by name.
Tool results are truncated to MAX_RESULT_CHARS to save tokens on the
next round of the conversation.

NOTE (vendored): the voice-action tools from the source (join_voice,
leave_voice, say_in_vc, mute_self, deafen_self, move_voice,
speak_in_stage) are pruned — only the read-only voice tools remain.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Awaitable

from src.actions.tools.context import ToolContext
from src.actions.utils.logger import logger

import src.actions.tools.channels as channels
import src.actions.tools.members as members
import src.actions.tools.messages as messages
import src.actions.tools.messaging as messaging
import src.actions.tools.reactions as reactions
import src.actions.tools.stickers_gifs as stickers_gifs
import src.actions.tools.voice as voice
import src.actions.tools.roles as roles
import src.actions.tools.profile as profile
import src.actions.tools.slash_commands as slash_commands
import src.actions.tools.bump_manager as bump_manager
import src.actions.tools.scheduler as scheduler_tools
import src.actions.tools.prefs as prefs

ToolFn = Callable[..., Awaitable[Any]]

# Max chars of a tool result fed back to the model (saves tokens).
MAX_RESULT_CHARS = 3000


# ============================================================
#  FULL schemas (command / event mode)
# ============================================================
TOOL_SCHEMAS: list[dict] = [
    {"type": "function", "function": {
        "name": "list_channels", "description": "List all channels, optionally filtered by type (text/voice/stage/thread/forum/category).",
        "parameters": {"type": "object", "properties": {"type_filter": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "search_channels", "description": "Fuzzy-search channel names. Returns ranked matches with scores.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_channel_details", "description": "Get full details (topic, nsfw, slowmode, overwrites, bitrate) of a channel by fuzzy name.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_channel_mention", "description": "Get the #channel mention string for a fuzzy name.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_channel_link", "description": "Get a clickable jump URL for a channel by fuzzy name.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "search_members", "description": "Fuzzy-search members by display name or username.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_member_details", "description": "Get full details (id, username, roles, bio, status) of a member by fuzzy name.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_member_mention", "description": "Get the @user mention string for a fuzzy member name.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_member_count", "description": "Get total member count and breakdown by status (online/idle/dnd/offline).",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_online_members", "description": "List members currently online (online/idle/dnd).",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_recent_joins", "description": "List members who joined the server most recently.",
        "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "get_recent_leaves", "description": "List members who recently left the server.",
        "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "get_recent_messages", "description": "Read latest N messages of a text channel. Embeds as JSON, images described via vision.",
        "parameters": {"type": "object", "properties": {"channel_query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["channel_query"]}}},
    {"type": "function", "function": {
        "name": "get_message_by_link", "description": "Fetch a single message by its Discord jump link.",
        "parameters": {"type": "object", "properties": {"link": {"type": "string"}}, "required": ["link"]}}},
    {"type": "function", "function": {
        "name": "send_message", "description": "Send a message to a channel. Use 'here' or 'this' for current channel. Can ping users, ping roles, and mention channels by fuzzy name.",
        "parameters": {"type": "object", "properties": {"channel_query": {"type": "string"}, "content": {"type": "string"}, "ping_users": {"type": "array", "items": {"type": "string"}}, "ping_roles": {"type": "array", "items": {"type": "string"}}, "mention_channels": {"type": "array", "items": {"type": "string"}}, "reply_to_link": {"type": "string"}}, "required": ["channel_query", "content"]}}},
    {"type": "function", "function": {
        "name": "send_dm", "description": "Send a DM to a user by fuzzy name.",
        "parameters": {"type": "object", "properties": {"user_query": {"type": "string"}, "content": {"type": "string"}}, "required": ["user_query", "content"]}}},
    {"type": "function", "function": {
        "name": "react_to_message", "description": "Add an emoji reaction to a message by its jump link.",
        "parameters": {"type": "object", "properties": {"message_link": {"type": "string"}, "emoji": {"type": "string"}}, "required": ["message_link", "emoji"]}}},
    {"type": "function", "function": {
        "name": "react_to_user_latest", "description": "React with an emoji to the latest message from a user in a channel.",
        "parameters": {"type": "object", "properties": {"channel_query": {"type": "string"}, "user_query": {"type": "string"}, "emoji": {"type": "string"}}, "required": ["channel_query", "user_query", "emoji"]}}},
    {"type": "function", "function": {
        "name": "react_to_recent", "description": "React to the N most recent messages in a channel with an emoji (batch operation).",
        "parameters": {"type": "object", "properties": {"channel_query": {"type": "string"}, "emoji": {"type": "string"}, "count": {"type": "integer", "description": "Number of recent messages to react to (default 5)"}}, "required": ["channel_query", "emoji"]}}},
    {"type": "function", "function": {
        "name": "search_gifs", "description": "Search for GIFs by keyword. Returns gif URLs.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "send_gif", "description": "Search a gif and send its URL to a channel with optional caption, ping_users, ping_roles.",
        "parameters": {"type": "object", "properties": {"channel_query": {"type": "string"}, "query": {"type": "string"}, "caption": {"type": "string"}, "ping_users": {"type": "array", "items": {"type": "string"}}, "ping_roles": {"type": "array", "items": {"type": "string"}}}, "required": ["channel_query", "query"]}}},
    {"type": "function", "function": {
        "name": "trending_gifs", "description": "Get trending GIFs from Klipy.",
        "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "list_stickers", "description": "List the guild's custom stickers.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "send_sticker", "description": "Send a guild sticker by fuzzy name to a channel.",
        "parameters": {"type": "object", "properties": {"channel_query": {"type": "string"}, "sticker_query": {"type": "string"}}, "required": ["channel_query", "sticker_query"]}}},
    {"type": "function", "function": {
        "name": "list_voice_channels", "description": "List all voice + stage channels with member counts.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_voice_state", "description": "Report current voice connection and members in the channel.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_vc_text_chat", "description": "Read the text chat associated with a voice channel.",
        "parameters": {"type": "object", "properties": {"channel_query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["channel_query"]}}},
    # --- Roles ---
    {"type": "function", "function": {
        "name": "search_roles", "description": "Fuzzy-search roles by name. Returns role mentions for pinging.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_role_mention", "description": "Get the <@&role_id> mention for a fuzzy role name.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    # --- Voice state control ---
    {"type": "function", "function": {
        "name": "get_current_vc", "description": "Get the voice channel the bot is currently in (for sending text there).",
        "parameters": {"type": "object", "properties": {}}}},
    # --- Profile & presence ---
    {"type": "function", "function": {
        "name": "change_nickname", "description": "Change the bot's nickname in the guild.",
        "parameters": {"type": "object", "properties": {"nick": {"type": "string"}}, "required": ["nick"]}}},
    {"type": "function", "function": {
        "name": "change_status", "description": "Change online status. Options: online, idle, dnd, invisible.",
        "parameters": {"type": "object", "properties": {"status": {"type": "string"}}, "required": ["status"]}}},
    {"type": "function", "function": {
        "name": "change_custom_status", "description": "Set a custom status text under your name.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "emoji": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "change_bio", "description": "Change the bot's bio / about me.",
        "parameters": {"type": "object", "properties": {"bio": {"type": "string"}}, "required": ["bio"]}}},
    {"type": "function", "function": {
        "name": "get_my_profile", "description": "Get the bot's own profile info.",
        "parameters": {"type": "object", "properties": {}}}},
    # --- Batch media ---
    {"type": "function", "function": {
        "name": "send_multiple_gifs", "description": "Send multiple different gifs to a channel in one call.",
        "parameters": {"type": "object", "properties": {"channel_query": {"type": "string"}, "query": {"type": "string"}, "count": {"type": "integer"}}, "required": ["channel_query", "query", "count"]}}},
    {"type": "function", "function": {
        "name": "send_multiple_stickers", "description": "Send multiple random stickers to a channel in one call.",
        "parameters": {"type": "object", "properties": {"channel_query": {"type": "string"}, "count": {"type": "integer"}}, "required": ["channel_query", "count"]}}},
    # --- User voice state ---
    {"type": "function", "function": {
        "name": "get_user_voice_state", "description": "Check which voice channel a user is currently in. Use for 'join my vc'.",
        "parameters": {"type": "object", "properties": {"user_query": {"type": "string"}}, "required": ["user_query"]}}},
    # --- VC text chat ---
    {"type": "function", "function": {
        "name": "send_vc_text", "description": "Send a message to a voice channel's text chat. Supports pinging users.",
        "parameters": {"type": "object", "properties": {"channel_query": {"type": "string"}, "content": {"type": "string"}, "ping_users": {"type": "array", "items": {"type": "string"}}}, "required": ["channel_query", "content"]}}},
    # --- Bot identification ---
    {"type": "function", "function": {
        "name": "identify_bots", "description": "List all bot accounts in the server (for finding bump bots, mod bots, etc.).",
        "parameters": {"type": "object", "properties": {}}}},
    # --- Slash commands ---
    {"type": "function", "function": {
        "name": "list_slash_commands", "description": "List all available slash commands in a channel.",
        "parameters": {"type": "object", "properties": {"channel_query": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "use_slash_command", "description": "Use a slash command (e.g. /bump) in a channel. If multiple bots register the same command, pass application_id to target a specific bot.",
        "parameters": {"type": "object", "properties": {"command_name": {"type": "string"}, "channel_query": {"type": "string"}, "application_id": {"type": "string", "description": "Target a specific bot by its application ID (use list_slash_commands to find it)"}}, "required": ["command_name"]}}},
    # --- Message management ---
    {"type": "function", "function": {
        "name": "delete_message", "description": "Delete the bot's own message by link or message_id.",
        "parameters": {"type": "object", "properties": {"message_link": {"type": "string"}, "channel_query": {"type": "string"}, "message_id": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {
        "name": "delete_last_message", "description": "Delete the bot's last N messages in a channel.",
        "parameters": {"type": "object", "properties": {"channel_query": {"type": "string"}, "count": {"type": "integer"}}, "required": []}}},
    {"type": "function", "function": {
        "name": "edit_message", "description": "Edit the bot's own message.",
        "parameters": {"type": "object", "properties": {"message_link": {"type": "string"}, "channel_query": {"type": "string"}, "message_id": {"type": "string"}, "new_content": {"type": "string"}}, "required": ["new_content"]}}},
    # --- Bump management ---
    {"type": "function", "function": {
        "name": "find_bump_commands", "description": "Find all bump bots in a channel (detects /bump commands from server-promotion bots, excludes music bots).",
        "parameters": {"type": "object", "properties": {"channel_query": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "bump_with_bot", "description": "Bump the server with a specific bump bot by its application ID.",
        "parameters": {"type": "object", "properties": {"application_id": {"type": "string"}, "channel_query": {"type": "string"}}, "required": ["application_id"]}}},
    {"type": "function", "function": {
        "name": "bump_all", "description": "Bump the server with ALL detected bump bots in a channel.",
        "parameters": {"type": "object", "properties": {"channel_query": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "get_bump_status", "description": "Get bump cooldown status for all tracked bump bots.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "schedule_message", "description": "Schedule a message: delay_s = send after N seconds; interval_s>0 = repeat every N seconds until told to stop.",
        "parameters": {"type": "object", "properties": {"channel_query": {"type": "string"}, "content": {"type": "string"}, "delay_s": {"type": "number"}, "interval_s": {"type": "number"}, "label": {"type": "string"}}, "required": ["channel_query", "content"]}}},
    {"type": "function", "function": {
        "name": "list_scheduled", "description": "List active scheduled tasks.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "stop_scheduled", "description": "Stop scheduled task(s) by id/label/channel; empty = stop all. For 'stop it'/'stop sending'.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "set_preference", "description": "Save an owner preference permanently, e.g. key=welcome_channel value=general.",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}}, "required": ["key", "value"]}}},
    {"type": "function", "function": {
        "name": "get_preferences", "description": "List stored owner preferences.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "delete_preference", "description": "Delete an owner preference by key.",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}}},
]


# ============================================================
#  CHAT schemas — minimal descriptions, fewer tools (saves tokens)
# ============================================================
_CHAT_TOOL_SCHEMAS: list[dict] = [
    {"type": "function", "function": {
        "name": "search_channels", "description": "Find channels by name.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_channel_details", "description": "Get channel info (topic, settings, permissions).",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "search_members", "description": "Find members by name.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_member_mention", "description": "Get @mention for a member.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_member_count", "description": "Get member count and online stats.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_recent_joins", "description": "See who joined recently.",
        "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "get_online_members", "description": "List online members.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_recent_messages", "description": "Read recent messages in a channel (with embeds). Use 'here' for current channel.",
        "parameters": {"type": "object", "properties": {"channel_query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["channel_query"]}}},
    {"type": "function", "function": {
        "name": "get_message_by_link", "description": "Fetch a message by its Discord link.",
        "parameters": {"type": "object", "properties": {"link": {"type": "string"}}, "required": ["link"]}}},
    {"type": "function", "function": {
        "name": "send_message", "description": "Send a message to a channel. Use 'here' or 'this' for current channel. Can ping users, ping roles, mention channels. Pass reply_to_link to reply to a specific message.",
        "parameters": {"type": "object", "properties": {"channel_query": {"type": "string"}, "content": {"type": "string"}, "ping_users": {"type": "array", "items": {"type": "string"}}, "ping_roles": {"type": "array", "items": {"type": "string"}}, "mention_channels": {"type": "array", "items": {"type": "string"}}, "reply_to_link": {"type": "string", "description": "Discord message link to reply to"}}, "required": ["channel_query", "content"]}}},
    {"type": "function", "function": {
        "name": "react_to_message", "description": "React to a message by its link.",
        "parameters": {"type": "object", "properties": {"message_link": {"type": "string"}, "emoji": {"type": "string"}}, "required": ["message_link", "emoji"]}}},
    {"type": "function", "function": {
        "name": "react_to_recent", "description": "React to the N most recent messages in a channel (batch).",
        "parameters": {"type": "object", "properties": {"channel_query": {"type": "string"}, "emoji": {"type": "string"}, "count": {"type": "integer"}}, "required": ["channel_query", "emoji"]}}},
    {"type": "function", "function": {
        "name": "send_gif", "description": "Send a gif to a channel. Supports ping_users and ping_roles.",
        "parameters": {"type": "object", "properties": {"channel_query": {"type": "string"}, "query": {"type": "string"}, "caption": {"type": "string"}, "ping_users": {"type": "array", "items": {"type": "string"}}, "ping_roles": {"type": "array", "items": {"type": "string"}}}, "required": ["channel_query", "query"]}}},
    {"type": "function", "function": {
        "name": "trending_gifs", "description": "Get trending GIFs from Klipy.",
        "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "list_stickers", "description": "List server stickers.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "send_sticker", "description": "Send a sticker to a channel.",
        "parameters": {"type": "object", "properties": {"channel_query": {"type": "string"}, "sticker_query": {"type": "string"}}, "required": ["channel_query", "sticker_query"]}}},
    {"type": "function", "function": {
        "name": "get_current_vc", "description": "Get the VC the bot is currently in.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "search_roles", "description": "Find roles by name for pinging.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "change_nickname", "description": "Change your nickname.",
        "parameters": {"type": "object", "properties": {"nick": {"type": "string"}}, "required": ["nick"]}}},
    {"type": "function", "function": {
        "name": "change_status", "description": "Change online status (online/idle/dnd/invisible).",
        "parameters": {"type": "object", "properties": {"status": {"type": "string"}}, "required": ["status"]}}},
    {"type": "function", "function": {
        "name": "change_custom_status", "description": "Set custom status text.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "emoji": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "change_bio", "description": "Change your bio.",
        "parameters": {"type": "object", "properties": {"bio": {"type": "string"}}, "required": ["bio"]}}},
    {"type": "function", "function": {
        "name": "send_multiple_gifs", "description": "Send N different gifs to a channel.",
        "parameters": {"type": "object", "properties": {"channel_query": {"type": "string"}, "query": {"type": "string"}, "count": {"type": "integer"}}, "required": ["channel_query", "query", "count"]}}},
    {"type": "function", "function": {
        "name": "send_multiple_stickers", "description": "Send N random stickers to a channel.",
        "parameters": {"type": "object", "properties": {"channel_query": {"type": "string"}, "count": {"type": "integer"}}, "required": ["channel_query", "count"]}}},
    {"type": "function", "function": {
        "name": "get_user_voice_state", "description": "Check which VC a user is in. Use for 'join my vc'.",
        "parameters": {"type": "object", "properties": {"user_query": {"type": "string"}}, "required": ["user_query"]}}},
    {"type": "function", "function": {
        "name": "send_vc_text", "description": "Send a message to a VC's text chat.",
        "parameters": {"type": "object", "properties": {"channel_query": {"type": "string"}, "content": {"type": "string"}, "ping_users": {"type": "array", "items": {"type": "string"}}}, "required": ["channel_query", "content"]}}},
    {"type": "function", "function": {
        "name": "identify_bots", "description": "List all bot accounts in the server.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "use_slash_command", "description": "Use a slash command like /bump. Pass application_id to target a specific bot when multiple bots have the same command.",
        "parameters": {"type": "object", "properties": {"command_name": {"type": "string"}, "channel_query": {"type": "string"}, "application_id": {"type": "string", "description": "Target a specific bot by application ID"}}, "required": ["command_name"]}}},
    {"type": "function", "function": {
        "name": "delete_message", "description": "Delete your own message by link or ID.",
        "parameters": {"type": "object", "properties": {"message_link": {"type": "string"}, "channel_query": {"type": "string"}, "message_id": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "delete_last_message", "description": "Delete your last N messages in a channel.",
        "parameters": {"type": "object", "properties": {"channel_query": {"type": "string"}, "count": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "edit_message", "description": "Edit your own message.",
        "parameters": {"type": "object", "properties": {"message_link": {"type": "string"}, "channel_query": {"type": "string"}, "message_id": {"type": "string"}, "new_content": {"type": "string"}}, "required": ["new_content"]}}},
    {"type": "function", "function": {
        "name": "find_bump_commands", "description": "Find all bump bots in a channel.",
        "parameters": {"type": "object", "properties": {"channel_query": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "bump_all", "description": "Bump with ALL bump bots in a channel.",
        "parameters": {"type": "object", "properties": {"channel_query": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "bump_with_bot", "description": "Bump with a specific bot by application_id.",
        "parameters": {"type": "object", "properties": {"application_id": {"type": "string"}, "channel_query": {"type": "string"}}, "required": ["application_id"]}}},
    {"type": "function", "function": {
        "name": "get_bump_status", "description": "Check bump cooldowns for all bots.",
        "parameters": {"type": "object", "properties": {}}}},
    # ---- Scheduled actions + owner preferences ----
    {"type": "function", "function": {
        "name": "schedule_message", "description": "Schedule a message: delay_s = first send after N seconds; interval_s>0 = repeat every N seconds forever until told to stop (owner only, survives restarts).",
        "parameters": {"type": "object", "properties": {"channel_query": {"type": "string"}, "content": {"type": "string"}, "delay_s": {"type": "number"}, "interval_s": {"type": "number"}, "label": {"type": "string"}}, "required": ["channel_query", "content"]}}},
    {"type": "function", "function": {
        "name": "list_scheduled", "description": "List all active scheduled/timed tasks (owner only).",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "stop_scheduled", "description": "Stop scheduled task(s) by id/label/channel — empty query stops ALL. Use when owner says 'stop it'/'stop sending' (owner only).",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "set_preference", "description": "Save an owner preference that persists forever, e.g. key=welcome_channel value=general (owner only).",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}}, "required": ["key", "value"]}}},
    {"type": "function", "function": {
        "name": "get_preferences", "description": "List stored owner preferences.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "delete_preference", "description": "Delete an owner preference by key.",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}}},
    {"type": "function", "function": {
        "name": "cleanup_my_messages", "description": "Delete the bot's own stale messages in a channel (older than 6h or beyond 15).",
        "parameters": {"type": "object", "properties": {"channel_query": {"type": "string"}}, "required": ["channel_query"]}}},
]


# ============================================================
#  Dispatch map (shared by both modes)
# ============================================================
_DISPATCH: dict[str, ToolFn] = {
    "list_channels": channels.list_channels,
    "search_channels": channels.search_channels,
    "get_channel_details": channels.get_channel_details,
    "get_channel_mention": channels.get_channel_mention,
    "get_channel_link": channels.get_channel_link,
    "search_members": members.search_members,
    "get_member_details": members.get_member_details,
    "get_member_mention": members.get_member_mention,
    "get_member_count": members.get_member_count,
    "get_online_members": members.get_online_members,
    "get_recent_joins": members.get_recent_joins,
    "get_recent_leaves": members.get_recent_leaves,
    "get_recent_messages": messages.get_recent_messages,
    "get_message_by_link": messages.get_message_by_link,
    "send_message": messaging.send_message,
    "send_dm": messaging.send_dm,
    "react_to_message": reactions.react_to_message,
    "react_to_user_latest": reactions.react_to_user_latest,
    "react_to_recent": reactions.react_to_recent,
    "search_gifs": stickers_gifs.search_gifs,
    "send_gif": stickers_gifs.send_gif,
    "trending_gifs": stickers_gifs.trending_gifs,
    "list_stickers": stickers_gifs.list_stickers,
    "send_sticker": stickers_gifs.send_sticker,
    "list_voice_channels": voice.list_voice_channels,
    "get_voice_state": voice.get_voice_state,
    "get_vc_text_chat": voice.get_vc_text_chat,
    "get_current_vc": voice.get_current_vc,
    "get_user_voice_state": voice.get_user_voice_state,
    "send_vc_text": voice.send_vc_text,
    # Roles
    "search_roles": roles.search_roles,
    "get_role_mention": roles.get_role_mention,
    # Profile & presence
    "change_nickname": profile.change_nickname,
    "change_status": profile.change_status,
    "change_custom_status": profile.change_custom_status,
    "change_bio": profile.change_bio,
    "get_my_profile": profile.get_my_profile,
    # Batch media
    "send_multiple_gifs": stickers_gifs.send_multiple_gifs,
    "send_multiple_stickers": stickers_gifs.send_multiple_stickers,
    # Bot identification
    "identify_bots": members.identify_bots,
    # Slash commands
    "list_slash_commands": slash_commands.list_slash_commands,
    "use_slash_command": slash_commands.use_slash_command,
    # Message management
    "delete_message": messaging.delete_message,
    "delete_last_message": messaging.delete_last_message,
    "edit_message": messaging.edit_message,
    "cleanup_my_messages": messaging.cleanup_my_messages,
    # Bump management
    "find_bump_commands": bump_manager.find_bump_commands,
    "bump_with_bot": bump_manager.bump_with_bot,
    "bump_all": bump_manager.bump_all,
    "get_bump_status": bump_manager.get_bump_status,
    # Scheduled actions + owner preferences
    "schedule_message": scheduler_tools.schedule_message,
    "list_scheduled": scheduler_tools.list_scheduled,
    "stop_scheduled": scheduler_tools.stop_scheduled,
    "set_preference": prefs.set_preference,
    "get_preferences": prefs.get_preferences,
    "delete_preference": prefs.delete_preference,
}


# ============================================================
#  Tool categories (used by the intent router to ship a filtered
#  tool set instead of all 56 — big token/latency win)
# ============================================================
_CATEGORIES: dict[str, str] = {
    "list_channels": "channels", "search_channels": "channels",
    "get_channel_details": "channels", "get_channel_mention": "channels",
    "get_channel_link": "channels",
    "search_members": "members", "get_member_details": "members",
    "get_member_mention": "members", "get_member_count": "members",
    "get_online_members": "members", "get_recent_joins": "members",
    "get_recent_leaves": "members", "identify_bots": "members",
    "get_recent_messages": "messages", "get_message_by_link": "messages",
    "send_message": "messaging", "send_dm": "messaging",
    "delete_message": "messaging", "delete_last_message": "messaging",
    "edit_message": "messaging", "cleanup_my_messages": "messaging",
    "react_to_message": "reactions", "react_to_user_latest": "reactions",
    "react_to_recent": "reactions",
    "search_gifs": "media", "send_gif": "media", "trending_gifs": "media",
    "list_stickers": "media", "send_sticker": "media",
    "send_multiple_gifs": "media", "send_multiple_stickers": "media",
    "list_voice_channels": "voice",
    "get_voice_state": "voice", "get_vc_text_chat": "voice",
    "get_current_vc": "voice", "get_user_voice_state": "voice",
    "send_vc_text": "voice",
    "search_roles": "roles", "get_role_mention": "roles",
    "change_nickname": "profile", "change_status": "profile",
    "change_custom_status": "profile", "change_bio": "profile",
    "get_my_profile": "profile",
    "list_slash_commands": "slash", "use_slash_command": "slash",
    "find_bump_commands": "bump", "bump_with_bot": "bump",
    "bump_all": "bump", "get_bump_status": "bump",
    "schedule_message": "scheduling", "list_scheduled": "scheduling",
    "stop_scheduled": "scheduling",
    "set_preference": "prefs", "get_preferences": "prefs",
    "delete_preference": "prefs",
}

# Always included — actions almost always need to resolve fuzzy names
# ("ping Mr Alien in #general") before they can execute.
_RESOLVER_TOOLS = {
    "search_channels", "search_members", "search_roles",
    "get_member_mention", "get_channel_mention", "get_role_mention",
    "get_user_voice_state", "get_current_vc",
}


def build_tools(mode: str = "command", categories: set | None = None) -> list[dict]:
    """Return tool schemas for a mode, optionally filtered to categories.

    categories=None -> legacy behaviour (full set for command/event, compact
    set for chat). categories={"messaging", ...} -> only matching tools plus
    the universal resolver tools.
    """
    if categories:
        return [
            s for s in TOOL_SCHEMAS
            if _CATEGORIES.get(s["function"]["name"]) in categories
            or s["function"]["name"] in _RESOLVER_TOOLS
        ]
    if mode == "chat":
        return _CHAT_TOOL_SCHEMAS
    return TOOL_SCHEMAS


def get_tool_names() -> list[str]:
    return list(_DISPATCH.keys())


async def dispatch(ctx: ToolContext, name: str, args: dict[str, Any]) -> str:
    """Run a tool by name. Returns a JSON string, truncated to save tokens."""
    # Groq sometimes hallucinates tool names with suffixes like
    # "search_channels<|channel|>commentary". Try to extract the base name.
    if name not in _DISPATCH:
        base = name.split("<")[0].split("|")[0].strip()
        if base in _DISPATCH:
            name = base
    fn = _DISPATCH.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    # Filter out empty-string keys that the model sometimes hallucinates.
    if args:
        args = {k: v for k, v in args.items() if k != "" and v != ""}
    try:
        result = await fn(ctx, **(args or {}))
        if isinstance(result, (str, bytes)):
            text = result if isinstance(result, str) else result.decode("utf-8", "replace")
        else:
            text = json.dumps(result, default=str, ensure_ascii=False)
    except TypeError as e:
        logger.error(f"Tool {name} arg error: {e}")
        return json.dumps({"error": f"Bad arguments for {name}: {e}"})
    except Exception as e:  # noqa: BLE001
        logger.exception(f"Tool {name} crashed")
        return json.dumps({"error": f"{type(e).__name__}: {e}"})
    # Truncate to save tokens on the next round.
    # If the result is a JSON object/list, try to truncate gracefully by
    # removing items rather than cutting mid-string (which produces invalid JSON).
    if len(text) > MAX_RESULT_CHARS:
        # Try graceful truncation for list-containing results.
        if isinstance(result, dict):
            for key in ("messages", "bump_commands", "members", "channels", "roles", "results"):
                if key in result and isinstance(result[key], list) and len(result[key]) > 1:
                    # Remove items from the BEGINNING (oldest) until we fit,
                    # keeping the most recent items (which usually matter more).
                    while len(result[key]) > 1 and len(json.dumps(result, default=str, ensure_ascii=False)) > MAX_RESULT_CHARS:
                        result[key].pop(0)
                    result[f"_{key}_truncated"] = True
                    text = json.dumps(result, default=str, ensure_ascii=False)
                    break
        if len(text) > MAX_RESULT_CHARS:
            # Last resort: truncate the string but keep valid JSON.
            text = json.dumps({"truncated": True, "preview": text[:MAX_RESULT_CHARS - 50] + "..."})
    return text
