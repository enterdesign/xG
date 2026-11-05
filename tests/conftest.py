import pytest

from xg import config as C


@pytest.fixture(autouse=True)
def _isolated_output(tmp_path, monkeypatch):
    """Testy nigdy nie pisza do prawdziwego predictions/ ani web/."""
    monkeypatch.setattr(C, "OUT", tmp_path)
    monkeypatch.setattr(C, "PRED", tmp_path / "predictions")
    monkeypatch.setattr(C, "WEB", tmp_path / "web")
