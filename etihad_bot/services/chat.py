"""High-level chat entry points used by the API and the evaluation harness."""
import uuid

from langchain_core.messages import HumanMessage

from etihad_bot.graph.workflow import get_app, get_checkpointer
from etihad_bot.services import store


class NotOwner(PermissionError):
    """The conversation exists and belongs to someone else."""


def check_owner(user_id: str, conversation_id: str) -> None:
    owner = store.conversation_owner(conversation_id)
    if owner is not None and owner != user_id:
        raise NotOwner(conversation_id)


def chat(user_id: str, message: str, conversation_id: str | None = None, app=None) -> dict:
    conversation_id = conversation_id or uuid.uuid4().hex
    check_owner(user_id, conversation_id)
    app = app or get_app()
    out = app.invoke(
        {"messages": [HumanMessage(content=message)], "user_id": user_id,
         "conversation_id": conversation_id},
        config={"configurable": {"thread_id": conversation_id}})
    return {"conversation_id": conversation_id, "message_id": out["message_id"],
            "answer": out["answer"], "status": out["status"], "intent": out.get("intent"),
            "citations": out.get("citations", []), "chunks": out.get("chunks", []),
            "rewritten_query": out["rewritten_query"]}


def history(user_id: str, conversation_id: str) -> list[dict]:
    """Resume: reload the checkpointed message list for a conversation the user owns."""
    check_owner(user_id, conversation_id)
    snap = get_app().get_state({"configurable": {"thread_id": conversation_id}})
    msgs = (snap.values or {}).get("messages", [])
    return [{"role": "user" if m.type == "human" else "assistant", "content": m.content}
            for m in msgs]


def delete_conversation(user_id: str, conversation_id: str) -> bool:
    """Delete messages, feedback and the LangGraph checkpoint. False if it doesn't exist."""
    if store.conversation_owner(conversation_id) is None:
        return False
    check_owner(user_id, conversation_id)
    store.delete_conversation_data(conversation_id)
    get_checkpointer().delete_thread(conversation_id)
    return True
