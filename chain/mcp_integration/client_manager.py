"""Request-scoped sessions and policy-filtered capabilities using the official SDK."""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta

import httpx
from jsonschema.validators import validator_for
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from pydantic import AnyUrl

from .config import MCPConfig, ServerConfig
from .identity import Identity

logger = logging.getLogger("rag.mcp")


def audit(request_id: str, event: str, **fields):
    # No arguments, result bodies, credentials, server URLs or exception messages.
    logger.info(
        json.dumps({"request_id": request_id, "event": event, **fields}, sort_keys=True)
    )


def _has_ref(value):
    if isinstance(value, dict):
        return "$ref" in value or any(_has_ref(v) for v in value.values())
    return isinstance(value, list) and any(_has_ref(v) for v in value)


@dataclass
class Capability:
    alias: str
    server_id: str
    original: str
    kind: str
    description: str
    schema: dict
    validator: object
    session: ClientSession

    def definition(self):
        return {
            "type": "function",
            "function": {
                "name": self.alias,
                "description": self.description,
                "parameters": self.schema,
            },
        }


@asynccontextmanager
async def connect(server: ServerConfig, config: MCPConfig):
    # Enter and exit all SDK contexts in the same task (AnyIO cancel scopes).
    async with AsyncExitStack() as stack:
        if server.transport == "stdio":
            child_env = {key: os.environ[source] for key, source in server.env.items()}
            stderr = stack.enter_context(open(os.devnull, "w"))  # noqa: ASYNC230, SIM115 -- managed by ExitStack; local device only
            # SDK adds a small platform-default environment, never all of os.environ.
            read, write = await stack.enter_async_context(
                stdio_client(
                    StdioServerParameters(
                        command=server.command,
                        args=server.args,
                        env=child_env,
                    ),
                    errlog=stderr,
                )
            )
        else:
            headers = {}
            if server.bearer_token_env:
                token = os.environ[server.bearer_token_env]
                if not token:
                    raise ValueError("Empty upstream credential")
                headers["Authorization"] = f"Bearer {token}"
            http = await stack.enter_async_context(
                httpx.AsyncClient(
                    headers=headers,
                    timeout=httpx.Timeout(config.timeout_seconds),
                    follow_redirects=False,
                    trust_env=False,
                )
            )
            read, write, _ = await stack.enter_async_context(
                streamable_http_client(
                    server.url,
                    http_client=http,
                )
            )
        session = await stack.enter_async_context(
            ClientSession(
                read,
                write,
                read_timeout_seconds=timedelta(seconds=config.timeout_seconds),
            )
        )
        async with asyncio.timeout(config.timeout_seconds):
            initialized = await session.initialize()
        yield session, initialized.capabilities


class ClientManager:
    def __init__(self, config: MCPConfig, identity: Identity, request_id: str):
        self.config, self.identity, self.request_id = config, identity, request_id
        self.capabilities: dict[str, Capability] = {}
        self.events: list[dict] = []
        self.sources: list[dict] = []

    def event(self, server: str, operation: str, status: str, **extra):
        event = {"server": server, "operation": operation, "status": status, **extra}
        self.events.append(event)
        audit(self.request_id, "mcp_operation", **event)

    async def _pages(self, method, field):
        items, cursor, seen = [], None, set()
        for _ in range(self.config.max_pages):
            async with asyncio.timeout(self.config.timeout_seconds):
                page = await method(cursor=cursor)
            if len(page.model_dump_json().encode()) > self.config.max_catalog_bytes:
                raise ValueError("Catalog page exceeds limit")
            items.extend(getattr(page, field))
            if len(items) > self.config.max_capabilities:
                raise ValueError("Catalog exceeds limit")
            cursor = page.nextCursor
            if not cursor:
                return items
            if cursor in seen:
                raise ValueError("Repeated pagination cursor")
            seen.add(cursor)
        raise ValueError("Catalog pagination exceeds limit")

    def _register(self, server, session, original, kind, description, schema):
        if len(self.capabilities) >= self.config.max_capabilities:
            raise ValueError("Too many exposed capabilities")
        if _has_ref(schema):
            raise ValueError("Referenced tool schemas are not supported")
        if schema.get("type") != "object":
            raise ValueError("Tool input schema must be an object")
        validator_class = validator_for(schema)
        validator_class.check_schema(schema)
        digest = hashlib.sha256(f"{server.id}:{kind}:{original}".encode()).hexdigest()[
            :12
        ]
        label = re.sub(r"[^a-zA-Z0-9_-]", "_", original)[:20]
        alias = f"{server.id}__{label}__{digest}"
        if alias in self.capabilities:
            raise ValueError("Duplicate capability")
        self.capabilities[alias] = Capability(
            alias,
            server.id,
            original,
            kind,
            f"[{server.id}/{kind}] {description[:1000]}",
            schema,
            validator_class(schema),
            session,
        )

    @asynccontextmanager
    async def open(self):
        async with AsyncExitStack() as stack:
            for server in self.config.servers:
                if not set(server.roles).intersection(self.identity.roles):
                    self.event(server.id, "connect", "denied")
                    continue
                if not server.read_only_tools and not server.resource_uris:
                    continue
                try:
                    session, supported = await stack.enter_async_context(
                        connect(server, self.config)
                    )
                    if supported.tools and server.read_only_tools:
                        for tool in await self._pages(session.list_tools, "tools"):
                            if tool.name in server.read_only_tools:
                                self._register(
                                    server,
                                    session,
                                    tool.name,
                                    "tool",
                                    tool.description or tool.name,
                                    tool.inputSchema,
                                )
                    if supported.resources and server.resource_uris:
                        for resource in await self._pages(
                            session.list_resources, "resources"
                        ):
                            uri = str(resource.uri)
                            if uri in server.resource_uris:
                                self._register(
                                    server,
                                    session,
                                    uri,
                                    "resource",
                                    f"Read {resource.name}. {resource.description or ''}",
                                    {
                                        "type": "object",
                                        "properties": {},
                                        "additionalProperties": False,
                                    },
                                )
                    self.event(server.id, "discover", "ok")
                except Exception:  # noqa: BLE001 -- isolate SDK/transport failures without exposing payloads
                    # A broken server cannot leave a partially accepted catalog active.
                    self.capabilities = {
                        k: v
                        for k, v in self.capabilities.items()
                        if v.server_id != server.id
                    }
                    self.event(server.id, "discover", "unavailable")
            yield self

    async def execute(self, alias: str, arguments: dict):
        capability = self.capabilities.get(alias)
        if capability is None:
            self.event("unknown", "call", "denied")
            return {"error": "Capability is not permitted"}, True
        started = time.monotonic()
        try:
            if len(json.dumps(arguments).encode()) > self.config.max_result_bytes:
                raise ValueError("Arguments exceed limit")
            capability.validator.validate(arguments)
        except Exception:  # noqa: BLE001 -- hostile argument/schema failure is a denied call
            self.event(
                capability.server_id,
                capability.kind,
                "invalid_arguments",
                capability=alias,
            )
            return {"error": "Arguments do not satisfy the permitted schema"}, True
        try:
            async with asyncio.timeout(self.config.timeout_seconds):
                if capability.kind == "resource":
                    result = await capability.session.read_resource(
                        AnyUrl(capability.original)
                    )
                    blocks, failed = result.contents, False
                else:
                    result = await capability.session.call_tool(
                        capability.original, arguments=arguments
                    )
                    blocks, failed = result.content, bool(result.isError)
            # Forward text/structured data only; never fetch links or embedded media.
            texts = [b.text for b in blocks if hasattr(b, "text")]
            structured = getattr(result, "structuredContent", None)
            body = {
                "server": capability.server_id,
                "capability": alias,
                "content": texts,
                "structured_content": structured,
                "is_error": failed,
            }
            if len(json.dumps(body).encode()) > self.config.max_result_bytes:
                self.event(
                    capability.server_id,
                    capability.kind,
                    "result_too_large",
                    capability=alias,
                )
                return {"error": "MCP result exceeds the configured size limit"}, True
            if not texts and structured is None:
                body["content"] = ["No supported text or structured content returned"]
            self.event(
                capability.server_id,
                capability.kind,
                "error" if failed else "ok",
                capability=alias,
                duration_ms=round((time.monotonic() - started) * 1000),
            )
            if not failed:
                self.sources.append(
                    {
                        "server": capability.server_id,
                        "kind": capability.kind,
                        "name": capability.original,
                        "capability": alias,
                        "content": body,
                    }
                )
            return body, failed
        except Exception:  # noqa: BLE001 -- sanitize upstream SDK/transport errors
            self.event(
                capability.server_id, capability.kind, "unavailable", capability=alias
            )
            return {"error": "MCP operation failed or timed out"}, True
