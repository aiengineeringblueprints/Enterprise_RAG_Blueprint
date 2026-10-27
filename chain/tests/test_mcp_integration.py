"""MCP boundary, orchestration, and real transport tests; no external LLM needed."""

import asyncio
import importlib.util
import socket
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from mcp import types
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcp_integration.client_manager import ClientManager
from mcp_integration.config import MCPConfig, ServerConfig
from mcp_integration.execution import MCPModelError, answer_with_mcp, run_tool_loop
from mcp_integration.identity import Identity, verify_identity

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "frontend_identity", ROOT / "frontend/authentication/mcp_identity.py"
)
signer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(signer)


@pytest.fixture
def identity_secret(monkeypatch):
    monkeypatch.setenv("MCP_IDENTITY_SECRET", "test-identity-key-" * 3)


def test_identity_round_trip_and_tampering(identity_secret):
    token = signer.sign_identity({"username": "alice", "roles": ["General"]})
    assert verify_identity(token) == Identity("alice", ("General",))
    with pytest.raises(HTTPException) as exc:
        verify_identity(token[:-1] + ("a" if token[-1] != "a" else "b"))
    assert exc.value.status_code == 401
    with (
        patch("mcp_integration.identity.time.time", return_value=99999999999),
        pytest.raises(HTTPException),
    ):
        verify_identity(token)


@pytest.mark.parametrize(
    "authorization", [None, "Bearer invalid", "Bearer a.b", "Basic test"]
)
def test_identity_rejects_invalid_tokens(identity_secret, authorization):
    with pytest.raises(HTTPException) as exc:
        verify_identity(authorization)
    assert exc.value.status_code == 401


def test_identity_missing_secret(monkeypatch):
    monkeypatch.delenv("MCP_IDENTITY_SECRET", raising=False)
    with pytest.raises(HTTPException) as exc:
        verify_identity(None)
    assert exc.value.status_code == 503


@pytest.mark.parametrize(
    "options",
    [
        {"url": "http://localhost/mcp"},
        {"url": "https://user:pass@localhost/mcp"},
        {"url": "file:///etc/passwd"},
        {"url": "https://localhost/mcp#fragment"},
        {"url": "https://localhost/mcp", "command": "/bin/sh"},
    ],
)
def test_http_config_rejects_unsafe_or_mixed_options(options):
    with pytest.raises(ValidationError):
        ServerConfig(id="demo", transport="streamable_http", **options)


def stdio_server(**updates):
    values = {
        "id": "demo",
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(ROOT / "chain/mcp_demo/server.py"), "--stdio"],
        "roles": ["General"],
        "read_only_tools": ["add", "get_retention_policy"],
        "resource_uris": ["blueprint://policy/retention"],
    }
    values.update(updates)
    return ServerConfig(**values)


def test_duplicate_server_ids_and_relative_executable_rejected():
    with pytest.raises(ValidationError):
        MCPConfig(servers=[stdio_server(), stdio_server()])
    with pytest.raises(ValidationError):
        stdio_server(command="python")


def manager(config=None, roles=("General",)):
    return ClientManager(
        config or MCPConfig(), Identity("alice", roles), "test-request"
    )


@pytest.mark.asyncio
async def test_denied_server_is_never_connected():
    host = manager(MCPConfig(servers=[stdio_server()]), roles=("Finance",))
    with patch("mcp_integration.client_manager.connect") as connect:
        async with host.open():
            assert not host.capabilities
    connect.assert_not_called()
    assert host.events[0]["status"] == "denied"


@pytest.mark.asyncio
async def test_real_stdio_discovery_validation_execution_and_resource():
    host = manager(MCPConfig(servers=[stdio_server()]))
    async with host.open():
        caps = list(host.capabilities.values())
        assert len(caps) == 3
        assert "slow_read" not in [c.original for c in caps]
        add = next(c for c in caps if c.original == "add")
        result, failed = await host.execute(add.alias, {"a": 2, "b": 3})
        assert not failed and "5" in result["content"][0]
        assert (await host.execute(add.alias, {"a": "bad", "b": 3}))[1]
        assert (await host.execute("unapproved", {}))[1]
        resource = next(c for c in caps if c.kind == "resource")
        result, failed = await host.execute(resource.alias, {})
        assert not failed and "30 days" in result["content"][0]
        assert len(host.sources) == 2
    # Transport/session contexts have exited without cross-task cancellation errors.


@pytest.mark.asyncio
async def test_real_stdio_timeout_and_oversized_result():
    host = manager(MCPConfig(servers=[stdio_server(read_only_tools=["slow_read"])]))
    async with host.open():
        host.config.timeout_seconds = 0.1
        slow = next(c for c in host.capabilities.values() if c.original == "slow_read")
        body, failed = await host.execute(slow.alias, {"seconds": 3})
        assert failed and "error" in body
    host = manager(MCPConfig(servers=[stdio_server()], max_result_bytes=256))
    async with host.open():
        policy = next(
            c
            for c in host.capabilities.values()
            if c.original == "get_retention_policy"
        )
        _, failed = await host.execute(policy.alias, {})
        assert failed and host.events[-1]["status"] == "result_too_large"


@pytest.mark.asyncio
async def test_parallel_request_sessions_do_not_share_roles_or_sources():
    async def query(roles):
        host = manager(MCPConfig(servers=[stdio_server()]), roles=roles)
        async with host.open():
            if host.capabilities:
                cap = next(c for c in host.capabilities.values() if c.original == "add")
                await host.execute(cap.alias, {"a": 1, "b": 2})
        return host

    allowed, denied = await asyncio.gather(query(("General",)), query(("Finance",)))
    assert len(allowed.sources) == 1
    assert not denied.sources and not denied.capabilities


@pytest.mark.asyncio
async def test_real_streamable_http(tmp_path):
    # A real local server with an OS-assigned port, no Docker or external network.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    script = tmp_path / "serve.py"
    script.write_text(
        f"import sys\nsys.path.insert(0, {str(ROOT / 'chain/mcp_demo')!r})\n"
        f"from server import mcp\nmcp.settings.host='127.0.0.1'\nmcp.settings.port={port}\n"
        "mcp.run(transport='streamable-http')\n"
    )
    process = subprocess.Popen(  # noqa: ASYNC220 -- test fixture, waited and terminated in finally
        [sys.executable, str(script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        async with httpx.AsyncClient(trust_env=False) as http:
            for _ in range(100):
                if process.poll() is not None:
                    pytest.fail("Demo HTTP server exited")
                try:
                    await http.get(f"http://127.0.0.1:{port}/mcp", timeout=0.2)
                    break
                except httpx.TransportError:
                    await asyncio.sleep(0.05)
            else:
                pytest.fail("Demo HTTP server did not start")
        server = ServerConfig(
            id="httpdemo",
            transport="streamable_http",
            url=f"http://127.0.0.1:{port}/mcp",
            allow_plain_http=True,
            roles=["General"],
            read_only_tools=["add"],
        )
        host = manager(MCPConfig(servers=[server]))
        async with host.open():
            cap = next(iter(host.capabilities.values()))
            body, failed = await host.execute(cap.alias, {"a": 4, "b": 5})
            assert not failed and "9" in body["content"][0]
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.asyncio
async def test_bounded_tool_loop_preserves_call_ids_and_forces_final_answer():
    host = manager(MCPConfig(max_calls=1))
    model = Mock()
    model.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(
                content="",
                tool_calls=[{"name": "forbidden", "args": {}, "id": "call-1"}],
            ),
            AIMessage(content="Unable to access that tool."),
        ]
    )
    messages = [HumanMessage(content="Question")]
    answer = await run_tool_loop(model, messages, host)
    assert answer.content == "Unable to access that tool."
    tool_message = next(m for m in messages if isinstance(m, ToolMessage))
    assert tool_message.tool_call_id == "call-1" and tool_message.status == "error"
    assert model.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_malformed_model_response_fails_clearly():
    model = Mock(
        ainvoke=AsyncMock(
            return_value=AIMessage(
                content="",
                invalid_tool_calls=[
                    {"name": "add", "args": "bad-json", "id": "x", "error": "invalid"}
                ],
            )
        )
    )
    with pytest.raises(MCPModelError):
        await run_tool_loop(model, [], manager())


@pytest.mark.asyncio
async def test_repeated_pagination_cursor_is_bounded():
    page = types.ListToolsResult(tools=[], nextCursor="same")
    listing = AsyncMock(return_value=page)
    with pytest.raises(ValueError, match="Repeated"):
        await manager()._pages(listing, "tools")
    assert listing.await_count == 2


def test_schema_refs_are_rejected_without_fetching():
    with pytest.raises(ValueError, match="Referenced"):
        manager()._register(
            stdio_server(),
            Mock(),
            "unsafe",
            "tool",
            "unsafe",
            {"type": "object", "$ref": "https://example.invalid/schema"},
        )


def test_mcp_endpoint_authenticates_before_any_tool_work(identity_secret, monkeypatch):
    from chain_api import app
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MCP_ENABLED", "true")
    with patch(
        "mcp_integration.execution.answer_with_mcp", new_callable=AsyncMock
    ) as run:
        response = TestClient(app).post(
            "/call_llm",
            json={"question": "test", "use_mcp": True, "user_roles": ["admin"]},
        )
    assert response.status_code == 401
    run.assert_not_called()


def test_mcp_disabled_and_verified_roles(identity_secret, monkeypatch):
    from chain_api import app
    from fastapi.testclient import TestClient

    client = TestClient(app)
    monkeypatch.setenv("MCP_ENABLED", "false")
    assert (
        client.post("/call_llm", json={"question": "x", "use_mcp": True}).status_code
        == 503
    )
    monkeypatch.setenv("MCP_ENABLED", "true")
    monkeypatch.setenv("MCP_CONFIG_FILE", str(ROOT / "chain/mcp_servers.example.json"))
    token = signer.sign_identity({"username": "alice", "roles": ["General"]})
    with patch(
        "mcp_integration.execution.answer_with_mcp",
        new_callable=AsyncMock,
        return_value={"result": "ok"},
    ) as run:
        response = client.post(
            "/call_llm",
            json={"question": "x", "use_mcp": True, "user_roles": ["admin"]},
            headers={"Authorization": token},
        )
    assert response.status_code == 200
    assert run.call_args.args[0].user_roles == ["General"]


@pytest.mark.asyncio
async def test_complete_rag_with_real_mcp_and_scripted_model():
    from chain_api import QuestionRequest
    from langchain_core.documents import Document
    from prompts.promt_manager import PromptKey

    model = Mock()
    bound = Mock()
    model.bind_tools.return_value = bound

    async def respond(messages):
        if not any(isinstance(m, ToolMessage) for m in messages):
            alias = model.bind_tools.call_args.args[0][0]["function"]["name"]
            return AIMessage(
                content="", tool_calls=[{"name": alias, "args": {}, "id": "policy-1"}]
            )
        assert "30 days" in messages[-1].content
        assert "uploaded.pdf" in str(messages)
        return AIMessage(
            content="The uploaded document and demo policy specify retention of 30 days [demo]."
        )

    bound.ainvoke = AsyncMock(side_effect=respond)
    retriever = Mock(
        ainvoke=AsyncMock(
            return_value=[
                Document(
                    page_content="Document policy",
                    metadata={"source": "uploaded.pdf", "category_bitmask": 1},
                ),
                Document(
                    page_content="Restricted content",
                    metadata={"source": "secret.pdf", "category_bitmask": 16},
                ),
            ]
        )
    )
    config = MCPConfig(
        servers=[
            stdio_server(read_only_tools=["get_retention_policy"], resource_uris=[])
        ]
    )
    request = QuestionRequest(
        question="What is the retention period?",
        prompt_key=PromptKey.SOURCE,
        used_model="openai",
    )
    with (
        patch("handle_llms.load_llm_model", return_value=model),
        patch("retriever.create_retriever", return_value=retriever),
        patch("query_refinement.refine_query", return_value=request.question),
    ):
        result = await answer_with_mcp(request, Identity("alice", ("General",)), config)
    assert "30 days" in result["result"]
    assert len(result["relevant_documents"]) == 2
    assert result["mcp"]["sources"][0]["server"] == "demo"
    assert "Restricted content" not in str(bound.ainvoke.call_args)


@pytest.mark.asyncio
async def test_mcp_path_retrieves_configured_top_k(monkeypatch):
    from chain_api import QuestionRequest
    from prompts.promt_manager import PromptKey

    monkeypatch.setenv("RETRIEVER_TOP_K", "9")
    model = Mock(ainvoke=AsyncMock(return_value=AIMessage(content="ok")))
    retriever = Mock(ainvoke=AsyncMock(return_value=[]))
    request = QuestionRequest(question="q", prompt_key=PromptKey.SOURCE, used_model="openai")
    with (
        patch("handle_llms.load_llm_model", return_value=model),
        patch("retriever.create_retriever", return_value=retriever) as create,
        patch("query_refinement.refine_query", return_value="q"),
    ):
        await answer_with_mcp(request, Identity("alice", ("General",)), MCPConfig(servers=[]))
    assert create.call_args.kwargs["returned_docs"] == 9


def test_document_policy_denies_unknown_tags_and_restricted_content():
    from langchain_core.documents import Document
    from mcp_integration.policy import permitted_documents

    docs = [
        Document(page_content=str(mask), metadata={"category_bitmask": mask})
        for mask in [1, 16, 17, 0, None, "1", True, 1024]
    ]
    assert [
        d.page_content
        for d in permitted_documents(docs, Identity("alice", ("General",)))
    ] == ["1"]
    assert [
        d.page_content
        for d in permitted_documents(
            docs, Identity("alice", ("General", "Finance & Controlling"))
        )
    ] == ["1", "16", "17"]
    assert not permitted_documents(docs, Identity("alice", ("unknown",)))


@pytest.mark.asyncio
async def test_unavailable_server_isolated_and_duplicate_tool_names_namespaced():
    host = manager(
        MCPConfig(
            servers=[
                stdio_server(id="broken", command="/nonexistent/blueprint-mcp-test"),
                stdio_server(id="first", read_only_tools=["add"], resource_uris=[]),
                stdio_server(id="second", read_only_tools=["add"], resource_uris=[]),
            ]
        )
    )
    async with host.open():
        assert len(host.capabilities) == 2
        assert len({cap.alias for cap in host.capabilities.values()}) == 2
        assert {cap.original for cap in host.capabilities.values()} == {"add"}
        assert any(
            e["server"] == "broken" and e["status"] == "unavailable"
            for e in host.events
        )
        cap = next(iter(host.capabilities.values()))
        cap.session.call_tool = AsyncMock(
            side_effect=ConnectionError("private upstream details")
        )
        body, failed = await host.execute(cap.alias, {"a": 1, "b": 2})
        assert failed and "private" not in str(body)


def test_mcp_endpoint_timeout_and_errors_are_sanitized(identity_secret, monkeypatch):
    from chain_api import app
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MCP_ENABLED", "true")
    token = signer.sign_identity({"username": "alice", "roles": ["General"]})
    client = TestClient(app)
    payload = {"question": "x", "use_mcp": True}
    headers = {"Authorization": token}
    with patch(
        "mcp_integration.config.load_config",
        return_value=MCPConfig(request_timeout_seconds=1),
    ):

        async def slow(*args):
            await asyncio.sleep(3)

        with patch("mcp_integration.execution.answer_with_mcp", side_effect=slow):
            assert (
                client.post("/call_llm", json=payload, headers=headers).status_code
                == 504
            )
        with patch(
            "mcp_integration.execution.answer_with_mcp",
            side_effect=RuntimeError("secret/path"),
        ):
            response = client.post("/call_llm", json=payload, headers=headers)
            assert response.status_code == 502 and "secret/path" not in response.text


def test_mcp_guardrail_inference_failure_denies(monkeypatch):
    from guardrails_manager import ensure_input_allowed

    monkeypatch.setenv("DISABLE_GUARDRAILS", "false")
    model = Mock()
    model.invoke.side_effect = RuntimeError("unavailable")
    with patch("handle_llms.load_llm_model", return_value=model):
        assert not ensure_input_allowed("question", fail_closed=True)[0]
