"""Raw-file discovery and RawSession lifecycle.

Pure logic (no QApplication). Source: ``file_model.list_raw_entries``
and ``session.RawSession``. NeXus/raw classification of opened files is
covered by the CopyWorker tests (it classifies inline, off the GUI
thread).
"""

from __future__ import annotations

import pytest

from mlgidlab import file_model
from mlgidlab.session import RawSession


def test_list_raw_entries_applies_size_and_ndim_filter(synthetic_raw):
    entries = file_model.list_raw_entries(synthetic_raw)
    assert len(entries) == 1
    e = entries[0]
    assert e.dataset_path == "raw/data0/image"
    assert e.shape == (4, 64, 64)


def test_raw_session_open_empty_list_raises_valueerror():
    with pytest.raises(ValueError):
        RawSession.open([])


def test_raw_session_open_missing_path_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        RawSession.open([tmp_path / "does_not_exist.h5"])


def test_raw_session_open_happy(synthetic_raw):
    session = RawSession.open([synthetic_raw])
    assert session.kind == "raw"
    assert session.raw_paths == [synthetic_raw.resolve()]
    assert session.temp_path == synthetic_raw.resolve()
