"""
PropEdge V16.0 — git_push.py
──────────────────────────────────────────────────────────────────────────────
Shared GitHub push module used by batch_predict.py and batch0_grade.py.

Uses the GitHub REST API (HTTPS + token) instead of SSH subprocess.
SSH git push times out in automated launchd environments because the SSH
agent is not forwarded. The REST API never has this problem.

HOW TO SET YOUR TOKEN:
  1. Go to github.com → Settings → Developer settings → Personal access tokens
  2. Generate a token with 'repo' scope (or 'contents: write' for fine-grained)
  3. Set it once in your terminal — it persists across sessions:
       python3 -c "
       import keyring
       keyring.set_password('propedge', 'github_token', 'YOUR_TOKEN_HERE')
       "
  4. Or set environment variable: export GITHUB_TOKEN=your_token_here
  5. Or add GITHUB_TOKEN = 'your_token_here' directly to config.py (less secure)

FILES PUSHED:
  - data/today.json        (daily predictions)
  - data/season_2025_26.json (graded plays)
  - data/dvp_rankings.json  (DVP cache)
  (game log CSVs are excluded — too large for GitHub API single-file push)
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

# GitHub repo details — extracted from GIT_REMOTE in config.py
GITHUB_OWNER = "iamwerewolf1007"
GITHUB_REPO  = "PropEdgeV16.0"
GITHUB_BRANCH = "main"

# Files pushed on every predict run (B1-B5) — small, fast
PUSH_FILES = [
    "data/today.json",
    "data/dvp_rankings.json",
]

# Files pushed after B0 grade — current month's file + index + today + dvp
# Monthly files replace the old single large season JSON (which was 84MB and caused push failures)
PUSH_FILES_GRADE = [
    "data/today.json",
    "data/dvp_rankings.json",
    # Monthly files added dynamically by push() when grade=True
]

# Files pushed after generate — all monthly files for both seasons
PUSH_FILES_GENERATE = [
    "data/today.json",
    "data/dvp_rankings.json",
    # Monthly files added dynamically by push() when generate=True
]


def _get_token() -> str | None:
    """
    Get GitHub token from (in order):
    1. macOS Keychain via keyring  — most reliable for launchd, survives reboots
            print("    3. Store it: python3 -c 'import keyring; keyring.set_password(\"propedge\", \"github_token\", \"YOUR_TOKEN\")'")
    2. GITHUB_TOKEN environment variable
    3. ROOT/.github_token file     — launchd-safe but fragile (file content errors)
    4. config.py GITHUB_TOKEN constant  — fallback hardcode

    RECOMMENDED: use the keychain (option 1). It never gets corrupted by paste errors.
    """
    # 1. macOS Keychain — most reliable, works in launchd without env vars
    try:
        import keyring
        token = keyring.get_password("propedge", "github_token")
        if token and token.strip().startswith("ghp_"):
            return token.strip()
    except Exception:
        pass

    # 2. Environment variable
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token and token.startswith("ghp_"):
        return token

    # 3. Token file — read and validate (must start with ghp_ to avoid paste errors)
    token_file = ROOT / ".github_token"
    if token_file.exists():
        try:
            raw = token_file.read_text().strip()
            # Only accept the first line (guards against multi-line paste accidents)
            token = raw.splitlines()[0].strip() if raw else ""
            if token and token.startswith("ghp_"):
                return token
            elif token:
                print(f"  ⚠ Git: .github_token exists but value looks wrong (doesn't start with ghp_)")
                print(f"     Got: {token[:20]}...")
        except Exception:
            pass

    # 4. config.py constant
    try:
        from config import GITHUB_TOKEN  # type: ignore
        if GITHUB_TOKEN and GITHUB_TOKEN.startswith("ghp_"):
            return GITHUB_TOKEN.strip()
    except ImportError:
        pass

    return None


def _ssl_context():
    """
    macOS Python (python.org installer) ships without system root certs linked.
    This causes CERTIFICATE_VERIFY_FAILED on GitHub API calls in launchd agents.

    Resolution order:
      1. certifi package (pip install certifi)  — most reliable
      2. System default context                 — works if certs are installed
    """
    try:
        import certifi
        import ssl
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        import ssl
        return ssl.create_default_context()


def _push_file(path: Path, token: str, message: str) -> bool:
    """Push a single file to GitHub via REST API. Returns True on success."""
    import urllib.request
    import urllib.error
    import ssl

    if not path.exists():
        return True  # nothing to push

    # Size check — GitHub REST API hard limit is 100MB, practical limit ~50MB
    file_size_mb = path.stat().st_size / 1024 / 1024
    if file_size_mb > 95:
        print(f"  ⚠ Git: {path.name} is {file_size_mb:.0f}MB — exceeds GitHub 100MB limit. Skipping.")
        print(f"     Run: python3 run.py generate  to rebuild with correct data (removes duplicates)")
        return False
    if file_size_mb > 50:
        print(f"  ⚠ Git: {path.name} is {file_size_mb:.0f}MB — large file, push may be slow...")

    try:
        content = path.read_bytes()
        b64 = base64.b64encode(content).decode()
    except Exception as e:
        print(f"  ⚠ Git: read error {path.name}: {e}")
        return False

    rel = path.relative_to(ROOT).as_posix()
    api_url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{rel}"

    headers = {
        "Authorization": f"token {token}",
        "Content-Type": "application/json",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "PropEdge-V16",
    }

    ctx = _ssl_context()

    # Get current SHA (needed to update existing file)
    sha = None
    try:
        req = urllib.request.Request(
            f"{api_url}?ref={GITHUB_BRANCH}",
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            data = json.loads(resp.read())
            sha = data.get("sha")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"  ⚠ Git: SHA fetch failed {path.name}: {e}")
            return False
        # 404 = new file, sha stays None
    except Exception as e:
        print(f"  ⚠ Git: SHA fetch error {path.name}: {e}")
        return False

    # Push the file
    payload: dict = {
        "message": message,
        "content": b64,
        "branch":  GITHUB_BRANCH,
    }
    if sha:
        payload["sha"] = sha

    try:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            api_url,
            data=body,
            headers=headers,
            method="PUT",
        )
        with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
            resp.read()
        return True
    except urllib.error.HTTPError as e:
        err = e.read().decode()[:200]
        print(f"  ⚠ Git: push failed {path.name}: {e.code} {err}")
        return False
    except Exception as e:
        print(f"  ⚠ Git: push error {path.name}: {e}")
        return False


def token_check() -> None:
    """
    Diagnostic: show exactly what token is being read and test it against GitHub.
    Run with: python3 run.py token-check
    """
    import urllib.request, urllib.error

    print("\n  PropEdge — GitHub Token Diagnostic")
    print("  " + "─" * 50)

    # Check each source
    sources = []

    try:
        import keyring
        t = keyring.get_password("propedge", "github_token")
        if t and t.strip().startswith("ghp_"):
            sources.append(("Keychain", t.strip()[:12] + "..."))
        elif t:
            sources.append(("Keychain", f"EXISTS but wrong format: {t.strip()[:20]}..."))
        else:
            sources.append(("Keychain", "NOT SET"))
    except ImportError:
        sources.append(("Keychain", "keyring not installed — run: pip3 install keyring"))

    env_t = os.environ.get("GITHUB_TOKEN", "").strip()
    sources.append(("Env GITHUB_TOKEN", (env_t[:12] + "...") if env_t.startswith("ghp_") else ("NOT SET" if not env_t else f"wrong format: {env_t[:20]}")))

    token_file = ROOT / ".github_token"
    if token_file.exists():
        raw = token_file.read_text().strip()
        first_line = raw.splitlines()[0].strip() if raw else ""
        if first_line.startswith("ghp_"):
            sources.append((".github_token", first_line[:12] + "..."))
        else:
            sources.append((".github_token", f"EXISTS but wrong: {first_line[:30]}"))
    else:
        sources.append((".github_token", "NOT FOUND"))

    for source, value in sources:
        sym = "✓" if "..." in value else "✗"
        print(f"  {sym} {source:<22} {value}")

    # Test the active token
    token = _get_token()
    print()
    if not token:
        print("  ✗ No valid token found in any source.")
        print()
        print("  Fix: run these commands one at a time:")
        print("    pip3 install keyring")
        print("    python3 -c \"import keyring; keyring.set_password('propedge', 'github_token', 'YOUR_TOKEN_HERE')\"")
        print("    python3 run.py token-check")
        return

    print(f"  Active token: {token[:12]}...{token[-4:]}")
    print(f"  Testing against GitHub API...")

    ctx = _ssl_context()
    api_url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "PropEdge-V16",
    }
    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            data = json.loads(resp.read())
            print(f"  ✓ Token VALID — repo: {data.get('full_name')} | private: {data.get('private')}")
            print(f"  ✓ Push should work. Run: python3 run.py predict 2")
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print(f"  ✗ Token REJECTED (401) — token has expired or been revoked.")
            print()
            print("  Fix:")
            print("    1. Go to github.com → Settings → Developer settings → Personal access tokens")
            print("    2. Generate new token → scope: repo → copy it")
            print("    3. Store it (one command, replace YOUR_TOKEN):")
            print("    3. Store it: python3 -c 'import keyring; keyring.set_password(\"propedge\", \"github_token\", \"YOUR_TOKEN\")'")
            print("    4. python3 run.py token-check")
        elif e.code == 404:
            print(f"  ✗ Repo not found (404) — check GITHUB_OWNER/GITHUB_REPO in git_push.py")
        else:
            print(f"  ✗ HTTP {e.code}: {e.read().decode()[:200]}")
    except Exception as e:
        print(f"  ✗ Connection error: {e}")


def push(message: str, files: list[str] | None = None, grade: bool = False, **kwargs) -> None:
    """
    Push files to GitHub via REST API.

    Args:
        message: Commit message
        files:   Explicit list of relative paths (overrides defaults)
        grade:   True  → PUSH_FILES_GRADE (today + season + dvp — used by B0)
                 False → PUSH_FILES       (today + dvp only — used by B1-B5)
    """
    token = _get_token()
    if not token:
        print("  ⚠ Git: No GITHUB_TOKEN found. Create a token file (launchd-safe):")
        print(f"    echo 'YOUR_TOKEN' > {ROOT / '.github_token'}")
        print(f"    chmod 600 {ROOT / '.github_token'}")
        print("    Or: export GITHUB_TOKEN=your_token_here")
        return

    # Build dynamic file list including monthly JSON files
    from monthly_split import get_push_paths
    if kwargs.get("generate"):
        # All monthly files for both seasons + today + dvp
        monthly_2526 = get_push_paths("2025_26", only_current_month=False)
        monthly_2425 = get_push_paths("2024_25", only_current_month=False)
        default_files = (PUSH_FILES_GENERATE or []) + monthly_2526 + monthly_2425
    elif grade:
        # Current month only + index + today + dvp (fast daily push)
        monthly_current = get_push_paths("2025_26", only_current_month=True)
        default_files = list(PUSH_FILES_GRADE) + monthly_current
    else:
        default_files = list(PUSH_FILES)
    targets = [ROOT / f for f in (files or default_files)]
    ok = 0; fail = 0; fail_names = []
    for fpath in targets:
        if not fpath.exists():
            continue
        success = _push_file(fpath, token, message)
        if not success:
            # Retry once after 3s — handles transient network errors
            # Skip retry for auth errors (401/403) — those need token fix
            import time as _t; _t.sleep(3)
            success = _push_file(fpath, token, message)
        if success:
            ok += 1
        else:
            fail += 1
            fail_names.append(fpath.name)

    existing = [t for t in targets if t.exists()]
    if fail_names:
        print(f"  ⚠ Git: pushed {ok}/{len(existing)} files — {fail} failed: {fail_names}")
    else:
        print(f"  ✓ Git: pushed {ok}/{len(existing)} files — {message}")
