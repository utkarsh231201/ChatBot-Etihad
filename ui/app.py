"""Streamlit chat UI. Run: streamlit run ui/app.py  (backend must be running on :8000)"""
import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Etihad Assistant", page_icon="✈️")
st.title("✈️ Etihad Policy & Flight Assistant")

ss = st.session_state
ss.setdefault("token", None)
ss.setdefault("conv_id", None)
ss.setdefault("msgs", [])
ss.setdefault("selected", "new")


def headers() -> dict:
    return {"Authorization": f"Bearer {ss.token}"}


def api(method: str, path: str, **kw):
    """Call the backend; if the session has expired, return to the sign-in screen."""
    resp = requests.request(method, f"{API}{path}", headers=headers(), timeout=120, **kw)
    if resp.status_code == 401:
        ss.token, ss.conv_id, ss.msgs, ss.selected = None, None, [], "new"
        ss.notice = "Your session expired. Please sign in again."
        st.rerun()
    return resp


def reset_chat():
    ss.conv_id, ss.msgs, ss.selected = None, [], "new"


# ---- sidebar: sign in / out --------------------------------------------------------------------
with st.sidebar:
    if ss.token is None:
        st.subheader("Sign in")
        st.caption("Sign in with your PNR to chat. Your conversations are saved so you can "
                   "reopen or delete them later.")
        pnr = st.text_input("PNR (format AB23FG, case-sensitive)", max_chars=6, placeholder="AB23FG")
        if st.button("Sign in", disabled=not pnr):
            try:
                r = requests.post(f"{API}/auth/login", json={"pnr": pnr}, timeout=15)
            except requests.RequestException as e:
                st.error(f"Backend not reachable: {e}")
            else:
                if r.ok:
                    ss.token = r.json()["token"]
                    reset_chat()
                    st.rerun()
                else:
                    st.error(r.json().get("detail", "Sign-in failed"))
    else:
        st.success("Signed in")
        if st.button("Sign out"):
            requests.post(f"{API}/auth/logout", headers=headers(), timeout=15)
            ss.token = None
            reset_chat()
            st.rerun()

        convs = api("GET", "/conversations").json()
        labels = {c["conversation_id"]: f"{c['first_question'][:35]} ({c['turns']})" for c in convs}
        ids = ["new"] + list(labels)
        if ss.selected not in ids:
            ss.selected = "new"
        choice = st.selectbox("Conversation", ids, index=ids.index(ss.selected),
                              format_func=lambda i: "➕ New conversation" if i == "new" else labels[i])
        if choice != ss.selected:
            ss.selected = choice
            if choice == "new":
                reset_chat()
            else:
                hist = api("GET", f"/conversations/{choice}").json()["messages"]
                ss.conv_id = choice
                ss.msgs = [dict(m, citations=[], message_id=None) for m in hist]
            st.rerun()

        if ss.selected != "new":
            st.divider()
            confirm = st.checkbox("I want to permanently delete this conversation")
            if st.button("🗑️ Delete conversation", disabled=not confirm):
                r = api("DELETE", f"/conversations/{ss.selected}")
                if r.ok:
                    reset_chat()
                    st.rerun()
                st.error(r.json().get("detail", "Could not delete"))


# ---- sign-in gate: nothing below runs until the user has signed in ---------------------------
if ss.token is None:
    if ss.pop("notice", None):
        st.warning("Your session expired. Please sign in again.")
    st.info("Please sign in with your PNR in the sidebar to start chatting. "
            "A PNR looks like **AB23FG**: 2 letters, 2 digits, 2 letters.")
    st.stop()


# ---- chat ------------------------------------------------------------------------------------------
def render(m, idx):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        for c in m.get("citations", []):
            if c.get("type") == "tool":
                st.caption(f"🔧 Source: MCP tool `{c['tool']}`")
            else:
                pg = f", p.{c['page']}" if c.get("page") else ""
                st.caption(f"[{c['ref']}] {c['document']} — {c['section']}{pg}")
        if m["role"] == "assistant" and m.get("message_id") and not m.get("rated"):
            up, down, _ = st.columns([1, 1, 8])
            for col, label, rating in ((up, "👍", 1), (down, "👎", -1)):
                if col.button(label, key=f"fb{rating}{idx}"):
                    api("POST", "/feedback", json={"message_id": m["message_id"], "rating": rating})
                    m["rated"] = True
                    st.rerun()


for i, m in enumerate(ss.msgs):
    render(m, i)

if prompt := st.chat_input("Ask about Etihad policies or flights…"):
    ss.msgs.append({"role": "user", "content": prompt})
    try:
        with st.spinner("Thinking…"):
            resp = api("POST", "/chat", json={"message": prompt, "conversation_id": ss.conv_id})
            resp.raise_for_status()
            r = resp.json()
    except requests.RequestException as e:
        ss.msgs.pop()
        st.error(f"Backend error: {e}. Start it with: uvicorn etihad_bot.api.main:app --port 8000")
        st.stop()
    ss.conv_id = r["conversation_id"]
    if ss.token:
        ss.selected = r["conversation_id"]
    ss.msgs.append({"role": "assistant", "content": r["answer"], "citations": r["citations"],
                    "message_id": r["message_id"]})
    st.rerun()
