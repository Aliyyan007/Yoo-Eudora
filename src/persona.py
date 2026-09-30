"""
Persona module — loads and manages the bot's personality configuration.
"""
from loguru import logger
from pydantic import BaseModel, Field
from typing import List, Optional


class Persona(BaseModel):
    """Validated persona configuration."""
    name: str = Field(..., description="The account name / display name (casual)")
    full_name: Optional[str] = Field(None, description="Full name — only revealed when asked")
    age: Optional[int] = Field(None, description="Age of the persona")
    gender: Optional[str] = Field(None, description="Gender of the persona")
    location: Optional[str] = Field(None, description="Where the persona lives")
    heritage: Optional[str] = Field(None, description="Cultural heritage / background")
    style: str = Field("casual", description="Communication style description")
    age_vibe: str = Field("young-adult", description="Age vibe for tone")
    interests: List[str] = Field(default_factory=list, description="Topics the persona is interested in")
    personality: str = Field("friendly", description="Personality traits")
    bio: Optional[str] = Field(None, description="Discord profile bio text")


def load_persona(config: dict) -> Persona:
    """Load and validate persona from config dict."""
    persona_data = config.get("persona", {})
    persona = Persona(**persona_data)
    logger.info(f"Loaded persona: {persona.name} | style={persona.style} | vibe={persona.age_vibe}")
    return persona


def persona_to_dict(persona: Persona) -> dict:
    """Convert Persona model to dict for use in prompts."""
    return persona.model_dump()
