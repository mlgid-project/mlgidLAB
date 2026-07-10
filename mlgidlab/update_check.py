"""Startup update check + post-update changelog (offline).

Two independent, best-effort features, both fail silently:

* ``latest_release`` asks the GitHub Releases API for the newest published
  release (pre-releases included — our alphas are pre-releases) and
  ``is_outdated`` compares it to the running version. Any network / parse
  failure returns ``None`` so the GUI simply shows nothing.
* ``whats_new`` reads the bundled ``CHANGELOG.md`` and returns the sections
  strictly newer than the last version the user ran (persisted by the host
  in ``QSettings``), so opening a freshly updated build can show what changed.

This module is deliberately Qt-free so the parsing / comparison logic is unit
testable without a QApplication; the host (``main_window``) runs
``latest_release`` off-thread and renders the results.
"""

from __future__ import annotations

import json
import logging
import sys
import urllib.error
import urllib.request
from pathlib import Path

from packaging.version import InvalidVersion, Version

logger = logging.getLogger("mlgidlab")

# Single source of truth for the GitHub repo. The repo was renamed from
# ``mlgidBASE_GUI`` to ``mlgidLAB``; GitHub still redirects the old name, but
# we use the canonical one everywhere (Releases API, releases page, and the
# git URL the install commands use) so nothing depends on that redirect.
_OWNER_REPO = "mlgid-project/mlgidLAB"
RELEASES_API = f"https://api.github.com/repos/{_OWNER_REPO}/releases"
RELEASES_PAGE = f"https://github.com/{_OWNER_REPO}/releases"

# Our own installed distribution + the git URL used to (re)install it (the
# same ``pip install "git+https://github.com/mlgid-project/mlgidLAB@vX"`` the
# README / getting_started / CHANGELOG document).
DIST_NAME = "mlgidlab"
REPO_GIT_URL = f"https://github.com/{_OWNER_REPO}"
# The distribution that anchors the heavy ``[pipeline]`` extra. If it is
# installed we keep the extra on self-update so the backend stays wired up.
_PIPELINE_ANCHOR_DIST = "mlgidbase"


def parse_version(tag: str | None) -> Version | None:
    """Parse a PEP 440 version from a tag like ``v0.1.0a9`` (``None`` if bad)."""
    if not tag:
        return None
    try:
        return Version(str(tag).strip().lstrip("vV"))
    except InvalidVersion:
        return None


def pick_latest_release(releases: object) -> tuple[str, str] | None:
    """From the GitHub ``/releases`` JSON list return ``(tag, html_url)`` of the
    highest PEP 440 version, skipping drafts.

    Pre-releases are kept (the project ships ``aN`` alphas). Returns ``None``
    when the list is empty or nothing parses as a version.
    """
    if not isinstance(releases, list):
        return None
    best: tuple[Version, str, str] | None = None
    for rel in releases:
        if not isinstance(rel, dict) or rel.get("draft"):
            continue
        tag = rel.get("tag_name") or rel.get("name") or ""
        version = parse_version(tag)
        if version is None:
            continue
        url = rel.get("html_url") or RELEASES_PAGE
        if best is None or version > best[0]:
            best = (version, tag, url)
    if best is None:
        return None
    return best[1], best[2]


def latest_release(timeout: float = 4.0) -> tuple[str, str] | None:
    """Fetch the latest release ``(tag, html_url)`` from GitHub.

    Returns ``None`` on any failure (offline, rate-limited, no releases
    published) so callers stay silent. Blocking — run off the GUI thread.
    """
    req = urllib.request.Request(
        RELEASES_API,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "mlgidLAB-update-check",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        logger.debug("update check: release fetch failed", exc_info=True)
        return None
    return pick_latest_release(data)


def is_outdated(current: str, latest: str) -> bool:
    """True when ``latest`` is a strictly newer PEP 440 version than
    ``current`` (False if either is unparsable)."""
    c, l = parse_version(current), parse_version(latest)
    if c is None or l is None:
        return False
    return l > c


# --------------------------------------------------------------------- #
# In-app self-update (pip install --upgrade from the GitHub tag)
# --------------------------------------------------------------------- #
def _read_direct_url(dist_name: str = DIST_NAME) -> dict | None:
    """Return the parsed ``direct_url.json`` of the installed ``dist_name``.

    ``direct_url.json`` records how a distribution was installed (PEP 610).
    Returns ``None`` when the distribution isn't found, has no such record
    (e.g. installed from a plain wheel), or the JSON can't be parsed — kept
    as its own helper so tests can monkeypatch it.
    """
    try:
        from importlib.metadata import distribution

        raw = distribution(dist_name).read_text("direct_url.json")
    except Exception:  # PackageNotFoundError, OSError, ...
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _dist_present(name: str) -> bool:
    """True when a distribution named ``name`` is importable metadata-wise."""
    try:
        from importlib.metadata import distribution

        distribution(name)
        return True
    except Exception:
        return False


def is_editable_install(dist_name: str = DIST_NAME) -> bool:
    """True when ``dist_name`` is an editable (``pip install -e``) install.

    Editable installs point at a live source checkout, so a
    ``pip install --upgrade`` would clobber the link rather than update it —
    the host uses this to refuse self-update for dev checkouts.
    """
    data = _read_direct_url(dist_name)
    if not data:
        return False
    dir_info = data.get("dir_info")
    return bool(isinstance(dir_info, dict) and dir_info.get("editable"))


def pipeline_extra_installed() -> bool:
    """True when the heavy ``[pipeline]`` extra looks installed.

    Detected via its anchor distribution (``mlgidbase``) so self-update can
    preserve the extra instead of silently dropping the backend.
    """
    return _dist_present(_PIPELINE_ANCHOR_DIST)


def self_update_supported(dist_name: str = DIST_NAME) -> tuple[bool, str]:
    """Whether an in-app ``pip install --upgrade`` is safe for this install.

    Returns ``(True, "")`` for a normal pip-managed install. Returns
    ``(False, reason)`` for an editable/dev checkout, or when the install
    can't be identified as a normal one (no ``direct_url.json``) — in both
    cases upgrading in place could break the environment, so the host offers
    only "View release" instead.
    """
    if is_editable_install(dist_name):
        return (
            False,
            "This is a development (editable) install; update it with "
            "'git pull' or 'pip install -e .' instead.",
        )
    if _read_direct_url(dist_name) is None:
        return (
            False,
            "Could not determine how mlgidLAB was installed; update it "
            "manually with pip.",
        )
    return True, ""


def build_upgrade_command(
    tag: str,
    *,
    executable: str | None = None,
    with_pipeline: bool | None = None,
) -> list[str]:
    """Build the ``pip install --upgrade`` argv for a given release ``tag``.

    Mirrors the documented install command
    (``pip install "mlgidlab[pipeline] @ git+…@<tag>"``). ``executable``
    defaults to the running interpreter so the upgrade lands in the active
    environment; ``with_pipeline`` defaults to whether the ``[pipeline]``
    extra is currently installed (preserving it across the upgrade).
    """
    if executable is None:
        executable = sys.executable
    if with_pipeline is None:
        with_pipeline = pipeline_extra_installed()
    extra = "[pipeline]" if with_pipeline else ""
    spec = f"{DIST_NAME}{extra} @ git+{REPO_GIT_URL}@{tag}"
    return [executable, "-m", "pip", "install", "--upgrade", spec]


# --------------------------------------------------------------------- #
# Changelog (offline, from the bundled CHANGELOG.md)
# --------------------------------------------------------------------- #
def _changelog_path() -> Path | None:
    """Locate the repo/package ``CHANGELOG.md`` (``None`` if absent)."""
    try:
        import mlgidlab

        p = Path(mlgidlab.__file__).resolve().parent.parent / "CHANGELOG.md"
        return p if p.is_file() else None
    except Exception:  # pragma: no cover - defensive
        logger.debug("update check: changelog path lookup failed", exc_info=True)
        return None


def parse_changelog_sections(text: str) -> list[tuple[Version | None, str, str]]:
    """Split a changelog into ``(version, heading, body)`` per ``## `` heading.

    ``version`` is parsed from the first token of the heading (e.g.
    ``0.1.0a9`` in ``## 0.1.0a9 — ninth alpha (2026-06-30)``) or ``None``.
    """
    sections: list[tuple[str, list[str]]] = []
    head: str | None = None
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if head is not None:
                sections.append((head, lines))
            head = line[3:].strip()
            lines = []
        elif head is not None:
            lines.append(line)
    if head is not None:
        sections.append((head, lines))
    out: list[tuple[Version | None, str, str]] = []
    for h, body_lines in sections:
        token = h.split()[0] if h.split() else ""
        out.append((parse_version(token), h, "\n".join(body_lines).strip()))
    return out


def whats_new(current: str, last_seen: str | None) -> list[tuple[str, str]] | None:
    """Changelog sections to show after an update: versions ``v`` with
    ``last_seen < v <= current`` (newest first).

    When ``last_seen`` is falsy (fresh install / first run ever) returns
    ``None`` — nothing is shown, the host just records the current version.
    Returns ``None`` when the changelog can't be read or nothing qualifies.
    """
    if not last_seen:
        return None
    path = _changelog_path()
    if path is None:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        logger.debug("update check: changelog read failed", exc_info=True)
        return None
    cur, low = parse_version(current), parse_version(last_seen)
    if cur is None or low is None or not (low < cur):
        return None
    picked: list[tuple[Version, str, str]] = []
    for version, head, body in parse_changelog_sections(text):
        if version is not None and low < version <= cur:
            picked.append((version, head, body))
    picked.sort(key=lambda t: t[0], reverse=True)
    return [(h, b) for _, h, b in picked] or None
