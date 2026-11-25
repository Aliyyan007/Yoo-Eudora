"""Action-engine configuration isolation and persona-only startup support."""
from src.action_engine.config.settings import Settings


def test_discord_token_optional_for_persona_only_layout(monkeypatch):
    monkeypatch.delenv("DISCORD_TOKEN", raising=False)
    monkeypatch.setenv("DISCORD_TOKEN_ISLA", "persona-token")
    s = Settings()
    assert s.discord_token == ""


def test_settings_do_not_hydrate_config_env_file_by_default(monkeypatch):
    # Even if config/.env contains a real token locally, direct Settings()
    # construction must rely on the process environment the host prepared.
    monkeypatch.delenv("DISCORD_TOKEN", raising=False)
    s = Settings()
    assert s.discord_token == ""


def test_owner_id_list_parsing(monkeypatch):
    monkeypatch.setenv("OWNER_USER_ID", "123,456;789")
    s = Settings()
    assert s.owner_ids == frozenset({123, 456, 789})


def test_owner_ids_env_fallback(monkeypatch):
    monkeypatch.delenv("OWNER_USER_ID", raising=False)
    monkeypatch.setenv("OWNER_IDS", "111,222")
    s = Settings()
    assert s.owner_ids == frozenset({111, 222})


def test_bump_channel_ids_json_and_csv_parsing(monkeypatch):
    monkeypatch.setenv("BUMP_CHANNEL_IDS", '[123, "456"]')
    assert Settings().bump_channel_ids == [123, 456]
    monkeypatch.setenv("BUMP_CHANNEL_IDS", "123,456")
    assert Settings().bump_channel_ids == [123, 456]
