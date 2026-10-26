"""Verify assertions from the trusted frontend, independently of request roles."""

import base64
import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import dataclass

from fastapi import HTTPException


@dataclass(frozen=True)
class Identity:
    subject: str
    roles: tuple[str, ...]


def verify_identity(authorization: str | None) -> Identity:
    secret = os.getenv("MCP_IDENTITY_SECRET", "")
    if len(secret) < 32:
        raise HTTPException(503, "MCP identity verification is not configured")
    try:
        if (
            not authorization
            or not authorization.startswith("Bearer ")
            or len(authorization) > 8192
        ):
            raise ValueError("Missing assertion")
        payload, signature = authorization[7:].split(".")
        expected = hmac.new(
            secret.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError("Invalid signature")
        claims = json.loads(
            base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
        )
        if not isinstance(claims, dict):
            raise TypeError("Invalid claims")
        now = time.time()
        if (
            claims.get("aud") != "rag-mcp"
            or not isinstance(claims.get("sub"), str)
            or not claims["sub"]
            or len(claims["sub"]) > 256
            or type(claims.get("iat")) is not int
            or type(claims.get("exp")) is not int
            or not now - 120 <= claims["iat"] <= now + 5
            or not now < claims["exp"] <= claims["iat"] + 120
            or not isinstance(claims.get("roles"), list)
            or not claims["roles"]
            or len(claims["roles"]) > 32
            or any(
                not isinstance(r, str) or not r or len(r) > 128 for r in claims["roles"]
            )
        ):
            raise ValueError("Invalid claims")
        return Identity(claims["sub"], tuple(claims["roles"]))
    except (ValueError, TypeError, KeyError):
        logging.getLogger("rag.mcp").warning("mcp_identity_denied")
        raise HTTPException(401, "Invalid or expired MCP identity assertion") from None
