import pytest
from resumini.db.sqlite import SQLiteStore


@pytest.fixture
def store(tmp_db):
    s = SQLiteStore(str(tmp_db))
    s.initialize()
    yield s
    s.close()


def test_add_and_get_materia(store):
    store.add_materia("civil", {"profesor": "Garcia", "cuatrimestre": "1C2026"})
    m = store.get_materia("civil")
    assert m["nombre"] == "civil"
    assert m["profesor"] == "Garcia"


def test_list_materias(store):
    store.add_materia("civil", {"profesor": "G"})
    store.add_materia("penal", {"profesor": "L"})
    materias = store.list_materias()
    assert len(materias) == 2


def test_update_preference(store):
    store.set_preference("formato", "bullet points")
    assert store.get_preference("formato") == "bullet points"


def test_get_preference_missing(store):
    assert store.get_preference("missing") is None


def test_add_session(store):
    store.add_session("s1", "resumir pdf civil", "2026-05-08T10:00:00")
    sessions = store.get_recent_sessions(limit=10)
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "s1"


def test_get_recent_sessions_limit(store):
    for i in range(5):
        store.add_session(f"s{i}", f"query {i}", "2026-05-08T10:00:00")
    sessions = store.get_recent_sessions(limit=3)
    assert len(sessions) == 3
