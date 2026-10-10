import pytest

from booksreview import config


@pytest.fixture(autouse=True)
def _trace_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRACE_DIR", tmp_path / "logs")
