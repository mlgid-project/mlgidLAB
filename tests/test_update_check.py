"""Startup update check + post-update changelog.

Covers the Qt-free logic in ``mlgidlab.update_check`` (version parsing,
release picking, changelog slicing, network fetch with a mocked urlopen)
and the MainWindow wiring (banner shown when outdated, up-to-date status
message, changelog dialog gated by the persisted last-seen version).

Source: update_check.py; main_window.py ``_on_update_check_finished`` /
``_maybe_show_changelog`` / ``_UpdateBanner``.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from mlgidlab import update_check as uc

pytestmark = pytest.mark.gui


# --------------------------- pure logic --------------------------- #
def test_parse_version_strips_v_and_rejects_junk():
    assert str(uc.parse_version("v0.1.0a9")) == "0.1.0a9"
    assert str(uc.parse_version("0.1.0a9")) == "0.1.0a9"
    assert uc.parse_version("not-a-version") is None
    assert uc.parse_version("") is None
    assert uc.parse_version(None) is None


def test_pick_latest_release_prefers_highest_and_skips_drafts():
    releases = [
        {"tag_name": "v0.1.0a8", "html_url": "u8"},
        {"tag_name": "v0.1.0a10", "html_url": "u10"},  # newest (alpha kept)
        {"tag_name": "v0.2.0", "html_url": "draft", "draft": True},  # skipped
        {"tag_name": "v0.1.0a9", "html_url": "u9"},
        {"name": "weird", "html_url": "bad"},  # unparsable -> ignored
    ]
    assert uc.pick_latest_release(releases) == ("v0.1.0a10", "u10")


def test_pick_latest_release_empty_and_non_list():
    assert uc.pick_latest_release([]) is None
    assert uc.pick_latest_release(None) is None
    assert uc.pick_latest_release({"tag_name": "v1"}) is None


def test_is_outdated():
    assert uc.is_outdated("0.1.0a9", "v0.1.0a10") is True
    assert uc.is_outdated("0.1.0a9", "v0.1.0a9") is False
    assert uc.is_outdated("0.1.0a10", "v0.1.0a9") is False
    assert uc.is_outdated("0.1.0a9", "garbage") is False


def test_parse_changelog_sections():
    text = (
        "# Changelog\n\nintro\n\n"
        "## 0.1.0a10 — tenth (2026-07-01)\n\nnew stuff\n\n"
        "## 0.1.0a9 — ninth (2026-06-30)\n\nold stuff\n"
    )
    secs = uc.parse_changelog_sections(text)
    assert [str(v) for v, _, _ in secs] == ["0.1.0a10", "0.1.0a9"]
    assert secs[0][2] == "new stuff"
    assert secs[1][1].startswith("0.1.0a9")


def test_whats_new_returns_only_newer_sections(tmp_path, monkeypatch):
    cl = tmp_path / "CHANGELOG.md"
    cl.write_text(
        "# Changelog\n\n"
        "## 0.1.0a10 — ten\n\nfeature ten\n\n"
        "## 0.1.0a9 — nine\n\nfeature nine\n\n"
        "## 0.1.0a8 — eight\n\nfeature eight\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(uc, "_changelog_path", lambda: cl)
    # updated a8 -> a10: show a10 and a9 (newest first), not a8
    out = uc.whats_new("0.1.0a10", "0.1.0a8")
    assert [h.split()[0] for h, _ in out] == ["0.1.0a10", "0.1.0a9"]
    # no update (same version) -> nothing
    assert uc.whats_new("0.1.0a10", "0.1.0a10") is None
    # fresh install (no last-seen) -> nothing
    assert uc.whats_new("0.1.0a10", None) is None
    # downgrade / last-seen newer -> nothing
    assert uc.whats_new("0.1.0a9", "0.1.0a10") is None


def test_whats_new_none_when_changelog_missing(monkeypatch):
    monkeypatch.setattr(uc, "_changelog_path", lambda: None)
    assert uc.whats_new("0.1.0a10", "0.1.0a8") is None


class _FakeResp:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_latest_release_parses_and_survives_offline(monkeypatch):
    payload = json.dumps(
        [{"tag_name": "v0.1.0a11", "html_url": "u11"}]
    ).encode()
    monkeypatch.setattr(
        uc.urllib.request, "urlopen",
        lambda req, timeout=None: _FakeResp(payload),
    )
    assert uc.latest_release() == ("v0.1.0a11", "u11")

    def _boom(req, timeout=None):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(uc.urllib.request, "urlopen", _boom)
    assert uc.latest_release() is None


# --------------------------- self-update logic --------------------------- #
def test_build_upgrade_command_with_and_without_pipeline():
    exe = "/some/python"
    plain = uc.build_upgrade_command(
        "v0.1.0a10", executable=exe, with_pipeline=False
    )
    assert plain == [
        exe, "-m", "pip", "install", "--upgrade",
        "mlgidlab @ git+https://github.com/mlgid-project/mlgidLAB@v0.1.0a10",
    ]
    full = uc.build_upgrade_command(
        "v0.1.0a10", executable=exe, with_pipeline=True
    )
    assert full[-1] == (
        "mlgidlab[pipeline] @ "
        "git+https://github.com/mlgid-project/mlgidLAB@v0.1.0a10"
    )


def test_build_upgrade_command_defaults_to_running_interpreter(monkeypatch):
    import sys
    monkeypatch.setattr(uc, "pipeline_extra_installed", lambda: False)
    cmd = uc.build_upgrade_command("v1.2.3")
    assert cmd[0] == sys.executable
    assert cmd[-1].endswith("@v1.2.3")


def test_is_editable_install(monkeypatch):
    monkeypatch.setattr(
        uc, "_read_direct_url",
        lambda dist_name=uc.DIST_NAME: {"dir_info": {"editable": True}},
    )
    assert uc.is_editable_install() is True
    monkeypatch.setattr(
        uc, "_read_direct_url",
        lambda dist_name=uc.DIST_NAME: {"dir_info": {"editable": False}},
    )
    assert uc.is_editable_install() is False
    # No direct_url.json at all (plain wheel install) -> not editable.
    monkeypatch.setattr(uc, "_read_direct_url", lambda dist_name=uc.DIST_NAME: None)
    assert uc.is_editable_install() is False


def test_self_update_supported(monkeypatch):
    # Editable -> unsupported with an explanatory reason.
    monkeypatch.setattr(
        uc, "_read_direct_url",
        lambda dist_name=uc.DIST_NAME: {"dir_info": {"editable": True}},
    )
    ok, reason = uc.self_update_supported()
    assert ok is False and "editable" in reason.lower()
    # Normal pip install (direct_url present, not editable) -> supported.
    monkeypatch.setattr(
        uc, "_read_direct_url",
        lambda dist_name=uc.DIST_NAME: {"url": "x", "dir_info": {"editable": False}},
    )
    assert uc.self_update_supported() == (True, "")
    # Unknown install (no direct_url) -> unsupported.
    monkeypatch.setattr(uc, "_read_direct_url", lambda dist_name=uc.DIST_NAME: None)
    ok, reason = uc.self_update_supported()
    assert ok is False and reason


def test_pipeline_extra_installed(monkeypatch):
    monkeypatch.setattr(uc, "_dist_present", lambda name: name == "mlgidbase")
    assert uc.pipeline_extra_installed() is True
    monkeypatch.setattr(uc, "_dist_present", lambda name: False)
    assert uc.pipeline_extra_installed() is False


# --------------------------- MainWindow wiring --------------------------- #
def test_banner_shown_only_when_outdated(main_window):
    from mlgidlab import __version__ as current

    assert main_window._update_banner.isHidden()
    # up to date -> banner stays hidden
    main_window._on_update_check_finished((f"v{current}", "http://x"))
    assert main_window._update_banner.isHidden()
    # newer available -> banner shown with the version
    main_window._on_update_check_finished(("v99.0.0", "http://x/rel"))
    assert not main_window._update_banner.isHidden()
    assert "99.0.0" in main_window._update_banner._label.text()


def test_up_to_date_status_message_only_when_requested(main_window):
    main_window._update_notify_uptodate = True
    main_window._on_update_check_finished(None)
    assert "up to date" in main_window.statusBar().currentMessage().lower()


def test_maybe_show_changelog_gated_by_last_seen(main_window, monkeypatch):
    from mlgidlab import __version__ as current
    from PySide6.QtCore import QSettings

    shown: list[list[tuple[str, str]]] = []
    monkeypatch.setattr(
        main_window, "_show_changelog_dialog",
        lambda cur, sections: shown.append(sections),
    )
    monkeypatch.setattr(
        uc, "whats_new", lambda cur, last: [("0.1.0aX — x", "body")]
    )
    QSettings().setValue(main_window._LAST_SEEN_VERSION_KEY, "0.0.1")
    main_window._maybe_show_changelog()
    assert shown  # dialog shown because last-seen was older
    # last-seen recorded as current so it won't show again
    assert str(QSettings().value(main_window._LAST_SEEN_VERSION_KEY)) == current

    # whats_new returns nothing -> no dialog, still records version
    shown.clear()
    monkeypatch.setattr(uc, "whats_new", lambda cur, last: None)
    main_window._maybe_show_changelog()
    assert not shown


def test_update_button_shown_only_when_self_update_supported(
    main_window, monkeypatch
):
    from PySide6.QtCore import QSettings

    QSettings().setValue(main_window._AUTO_UPDATE_KEY, False)
    # Supported install -> banner offers the "Update now" button.
    monkeypatch.setattr(uc, "self_update_supported", lambda *a, **k: (True, ""))
    main_window._on_update_check_finished(("v99.0.0", "http://x/rel"))
    assert not main_window._update_banner.isHidden()
    assert not main_window._update_banner._update_btn.isHidden()
    # Editable / unsupported -> banner shown but no in-app install button.
    monkeypatch.setattr(
        uc, "self_update_supported", lambda *a, **k: (False, "editable")
    )
    main_window._on_update_check_finished(("v99.0.0", "http://x/rel"))
    assert not main_window._update_banner.isHidden()
    assert main_window._update_banner._update_btn.isHidden()


def test_auto_update_installs_on_launch_when_enabled(main_window, monkeypatch):
    from PySide6.QtCore import QSettings

    requested: list[bool] = []
    monkeypatch.setattr(uc, "self_update_supported", lambda *a, **k: (True, ""))
    # Auto-update on launch goes through the same confirm-then-install flow as
    # the button (never a silent install), so it routes to
    # _on_install_update_requested, not _start_update_install directly.
    monkeypatch.setattr(
        main_window, "_on_install_update_requested",
        lambda: requested.append(True),
    )
    QSettings().setValue(main_window._AUTO_UPDATE_KEY, True)
    main_window._on_update_check_finished(("v99.0.0", "http://x/rel"))
    assert requested == [True]
    assert main_window._update_banner.isHidden()
    # Reset so the setting doesn't leak into other tests in this process.
    QSettings().setValue(main_window._AUTO_UPDATE_KEY, False)


def test_update_now_menu_confirms_install_when_newer(main_window, monkeypatch):
    # Help -> Update now… runs a check with install_when_found; on finding a
    # newer supported release it goes straight to the confirm dialog, not the
    # banner.
    confirmed: list[bool] = []
    monkeypatch.setattr(uc, "self_update_supported", lambda *a, **k: (True, ""))
    monkeypatch.setattr(
        main_window, "_on_install_update_requested",
        lambda: confirmed.append(True),
    )
    main_window._update_install_when_found = True
    main_window._on_update_check_finished(("v99.0.0", "http://x/rel"))
    assert confirmed == [True]
    assert main_window._update_banner.isHidden()


def test_auto_update_ignored_for_unsupported_install(main_window, monkeypatch):
    from PySide6.QtCore import QSettings

    requested: list[bool] = []
    # Toggle is on, but the install is editable -> never auto-install; the
    # banner is shown without the "Update now" button.
    monkeypatch.setattr(
        uc, "self_update_supported", lambda *a, **k: (False, "editable")
    )
    monkeypatch.setattr(
        main_window, "_on_install_update_requested",
        lambda: requested.append(True),
    )
    QSettings().setValue(main_window._AUTO_UPDATE_KEY, True)
    main_window._on_update_check_finished(("v99.0.0", "http://x/rel"))
    assert requested == []
    assert not main_window._update_banner.isHidden()
    assert main_window._update_banner._update_btn.isHidden()
    QSettings().setValue(main_window._AUTO_UPDATE_KEY, False)
