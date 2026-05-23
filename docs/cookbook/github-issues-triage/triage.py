"""GitHub issues fetcher for the github-issues-triage cookbook recipe."""

import json
import os
import urllib.request
from typing import Any


def fetch_issues_as_list() -> str:
    """Fetch open GitHub issues and return them as a JSON array of formatted strings.

    Reads GITHUB_TOKEN and GITHUB_REPO from the environment. Skips pull requests.
    Each item in the returned list is a formatted string containing the issue number,
    title, and a truncated body — ready for the factory node to iterate over.

    :raises ValueError: If GITHUB_TOKEN or GITHUB_REPO are not set.
    :returns: JSON-encoded list of formatted issue strings.
    """
    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPO", "")
    if not token or not repo:
        raise ValueError("GITHUB_TOKEN and GITHUB_REPO environment variables must be set")

    url = f"https://api.github.com/repos/{repo}/issues?state=open&per_page=10"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "sirenspec/0.1",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        issues: list[dict[str, Any]] = json.loads(resp.read())

    formatted = [
        f"#{issue['number']}: {issue['title']}\n{(issue.get('body') or '')[:400]}"
        for issue in issues
        if not issue.get("pull_request")
    ]
    return json.dumps(formatted)
