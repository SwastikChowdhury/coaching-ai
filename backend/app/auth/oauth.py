"""Google OAuth 2.0 (authorization-code flow).

Two responsibilities:
  1. Build the URL we redirect the browser to (get_google_auth_url).
  2. Exchange the code Google sends back for the user's profile
     (exchange_code_for_user): code -> access_token -> userinfo.

GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET are read from the environment. The
redirect URI is hard-coded to the local callback and must match exactly what is
registered in the Google Cloud console.
"""

import os
from urllib.parse import urlencode

import httpx
from dotenv import load_dotenv
from fastapi import HTTPException

load_dotenv()

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"
REDIRECT_URI = "http://localhost:8000/auth/google/callback"
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:5173")

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")


def get_google_auth_url() -> str:
    """Build the Google authorization URL the user is redirected to.

    access_type=offline requests a refresh token from Google (harmless even
    though we mint our own session tokens); the scopes cover the profile fields
    exchange_code_for_user later reads.
    """
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"


async def exchange_code_for_user(code: str) -> dict:
    """Exchange an auth code for the user's Google profile.

    Steps:
      1. POST the code to the token endpoint -> access_token.
      2. GET userinfo with that access_token -> profile.

    Returns the userinfo dict (id, email, given_name, family_name, ...). Raises
    HTTPException(400) if either call fails or no access_token comes back, so the
    callback endpoint can surface a clean error instead of a 500.
    """
    token_payload = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            token_resp = await client.post(GOOGLE_TOKEN_URL, data=token_payload)
            token_resp.raise_for_status()
        except httpx.HTTPError:
            raise HTTPException(status_code=400, detail="Failed to exchange Google code")

        access_token = token_resp.json().get("access_token")
        if not access_token:
            raise HTTPException(status_code=400, detail="No access token from Google")

        try:
            userinfo_resp = await client.get(
                GOOGLE_USERINFO_URL,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            userinfo_resp.raise_for_status()
        except httpx.HTTPError:
            raise HTTPException(status_code=400, detail="Failed to fetch Google profile")

    return userinfo_resp.json()
