import pytest

from etihad_bot.config import get_settings
from etihad_bot.services import auth, store


@pytest.fixture
def tmp_db(tmp_path):
    s = get_settings()
    old = s.sqlite_dir
    s.sqlite_dir = str(tmp_path)
    store._initialised.clear()
    yield
    s.sqlite_dir = old
    store._initialised.clear()


@pytest.mark.parametrize("pnr", ["AB23FG", "ZZ00ZZ", "QW99ER", "AB12CD"])
def test_valid_pnr(pnr):
    assert auth.validate_pnr(pnr) is None


@pytest.mark.parametrize("pnr", [
    "SADFER",    # all letters (the reported bug)
    "123456",    # all digits
    "ABC123",    # wrong shape: 3 letters + 3 digits
    "A1B2C3",    # mixed in the wrong places
    "AB2CFG",    # digit in the letter slot
    "AB23F1",    # digit at the end
    "1B23FG",    # digit at the start
    "ab23fg", "Ab23FG", "AB23Fg",   # case-sensitive
    "AB23F", "AB23FGH", "AB 3FG", "AB-3FG", "", " AB23FG", "AB23FG ",
])
def test_pnr_must_match_exact_format(pnr):
    assert auth.validate_pnr(pnr) is not None


def test_user_id_is_stable_and_hides_pnr():
    uid = auth.user_id_for_pnr("AB23FG")
    assert uid == auth.user_id_for_pnr("AB23FG") and "AB23FG" not in uid
    assert uid != auth.user_id_for_pnr("AB23FH")


def test_session_ownership_and_delete(tmp_db):
    token = store.create_session("pnr:u1")
    assert store.user_for_token(token) == "pnr:u1"
    assert store.user_for_token("bogus") is None
    store.delete_session(token)
    assert store.user_for_token(token) is None

    store.save_interaction(message_id="m1", user_id="pnr:u1", conversation_id="c1", question="q",
                           rewritten_query="q", intent="policy", answer="a", status="answered",
                           chunks=[], citations=[])
    assert store.conversation_owner("c1") == "pnr:u1"
    assert store.message_owner("m1") == "pnr:u1"
    assert store.save_feedback(message_id="m1", rating=1)
    assert store.delete_conversation_data("c1") == 1
    assert store.conversation_owner("c1") is None
    with store.conn() as c:
        assert c.execute("select count(*) from feedback").fetchone()[0] == 0

def test_api_requires_sign_in(tmp_db):
    from fastapi.testclient import TestClient

    from etihad_bot.api.main import app
    c = TestClient(app)
    body = {"message": "hello"}
    assert c.post("/chat", json=body).status_code == 401
    assert c.post("/chat", json=body, headers={"X-Guest-Id": "someguest0001"}).status_code == 401
    assert c.post("/chat", json=body, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert c.get("/conversations").status_code == 401
    assert c.post("/feedback", json={"message_id": "x", "rating": 1}).status_code == 401
    assert c.delete("/conversations/abc").status_code == 401
    assert c.get("/health").status_code == 200                                  # public
    assert c.post("/auth/login", json={"pnr": "SADFER"}).status_code == 422     # public, validated
    token = c.post("/auth/login", json={"pnr": "AB23FG"}).json()["token"]
    assert c.get("/conversations", headers={"Authorization": f"Bearer {token}"}).status_code == 200