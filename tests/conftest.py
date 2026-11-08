import pytest

from xg import config as C


@pytest.fixture(autouse=True)
def _isolated_output(tmp_path, monkeypatch):
    """Testy nigdy nie pisza do prawdziwego predictions/ ani web/."""
    monkeypatch.setattr(C, "OUT", tmp_path)
    monkeypatch.setattr(C, "PRED", tmp_path / "predictions")
    monkeypatch.setattr(C, "WEB", tmp_path / "web")


@pytest.fixture(autouse=True)
def _no_network_for_results(monkeypatch):
    """ESPN nie jest odpytywany w testach (chyba ze test poda wlasny fetch_json)."""
    from xg import freeresults as FR

    def boom(*a, **k):
        raise OSError("siec wylaczona w testach")
    monkeypatch.setattr(FR, "fetch_json", boom)
