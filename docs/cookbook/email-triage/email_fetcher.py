"""Gmail fetcher for the email-triage cookbook recipe."""

import json
import os
import urllib.request
from typing import Any


def fetch_latest_unread() -> str:
    """Fetch the most recent unread email from Gmail and return it as a formatted string.

    Reads GMAIL_ACCESS_TOKEN from the environment. Requires an OAuth 2.0 bearer
    token with the gmail.readonly scope.

    :raises ValueError: If GMAIL_ACCESS_TOKEN is not set.
    :returns: Formatted string containing the email sender, subject, and snippet.
    """
    token = os.environ.get("GMAIL_ACCESS_TOKEN", "")
    if not token:
        raise ValueError("GMAIL_ACCESS_TOKEN environment variable must be set")

    headers = {"Authorization": f"Bearer {token}"}

    list_url = "https://gmail.googleapis.com/gmail/v1/users/me/messages?labelIds=UNREAD&maxResults=1"
    req = urllib.request.Request(list_url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        data: dict[str, Any] = json.loads(resp.read())

    messages = data.get("messages", [])
    if not messages:
        return json.dumps({"from": "none", "subject": "none", "snippet": "No unread messages found."})

    msg_id = messages[0]["id"]
    msg_url = (
        f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{msg_id}"
        "?format=metadata&metadataHeaders=Subject&metadataHeaders=From"
    )
    req = urllib.request.Request(msg_url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        msg: dict[str, Any] = json.loads(resp.read())

    header_map = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
    return json.dumps({
        "from": header_map.get("From", "Unknown"),
        "subject": header_map.get("Subject", "No subject"),
        "snippet": msg.get("snippet", ""),
    })
