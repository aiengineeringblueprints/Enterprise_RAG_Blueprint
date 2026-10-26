"""Sign the authenticated Streamlit session for the chain's MCP boundary."""

import base64
import hashlib
import hmac
import json
import os
import time


def sign_identity(user_info: dict) -> str:
    secret = os.getenv("MCP_IDENTITY_SECRET", "")
    if len(secret) < 32 or not user_info.get("username") or not user_info.get("roles"):
        raise ValueError(
            "MCP requires a configured identity secret and an authenticated user with roles"
        )
    now = int(time.time())
    claims = {
        "sub": user_info["username"],
        "roles": user_info["roles"],
        "aud": "rag-mcp",
        "iat": now,
        "exp": now + 60,
    }
    payload = (
        base64.urlsafe_b64encode(json.dumps(claims, separators=(",", ":")).encode())
        .decode()
        .rstrip("=")
    )
    signature = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"Bearer {payload}.{signature}"
