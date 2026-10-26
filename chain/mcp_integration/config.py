"""Administrator-owned configuration. Never accept connection details from chat."""

import json
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,23}$")
    transport: Literal["stdio", "streamable_http"]
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)  # child variable -> host variable
    bearer_token_env: str | None = None
    roles: list[str] = Field(default_factory=list)
    # Explicit admin attestation that these tools are read-only. Annotations aren't trusted.
    read_only_tools: list[str] = Field(default_factory=list)
    resource_uris: list[str] = Field(default_factory=list)
    allow_plain_http: bool = False

    @model_validator(mode="after")
    def check_transport(self):
        if self.transport == "streamable_http":
            parts = urlsplit(self.url or "")
            schemes = {"https", "http"} if self.allow_plain_http else {"https"}
            if (
                parts.scheme not in schemes
                or not parts.hostname
                or parts.username
                or parts.password
                or parts.fragment
                or self.command
                or self.args
                or self.env
            ):
                raise ValueError(
                    "HTTP requires an approved URL without credentials/fragment or subprocess options"
                )
        elif (
            not self.command
            or not Path(self.command).is_absolute()
            or self.url
            or self.bearer_token_env
        ):
            raise ValueError(
                "stdio requires an absolute executable and no HTTP options"
            )
        return self


class MCPConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    servers: list[ServerConfig] = Field(default_factory=list, max_length=16)
    timeout_seconds: float = Field(default=15, ge=0.1, le=120)
    request_timeout_seconds: float = Field(default=90, ge=1, le=300)
    max_rounds: int = Field(default=4, ge=1, le=10)
    max_calls: int = Field(default=8, ge=1, le=32)
    max_result_bytes: int = Field(default=16000, ge=256, le=100000)
    max_catalog_bytes: int = Field(default=128000, ge=1024, le=1000000)
    max_capabilities: int = Field(default=64, ge=1, le=256)
    max_pages: int = Field(default=8, ge=1, le=32)

    @model_validator(mode="after")
    def unique_servers(self):
        if len({s.id for s in self.servers}) != len(self.servers):
            raise ValueError("Duplicate MCP server ID")
        return self


def enabled() -> bool:
    return os.getenv("MCP_ENABLED", "false").lower() in {"true", "1", "yes"}


def load_config() -> MCPConfig:
    path = Path(os.environ["MCP_CONFIG_FILE"])
    if path.stat().st_size > 256000:
        raise ValueError("MCP configuration is too large")
    return MCPConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
