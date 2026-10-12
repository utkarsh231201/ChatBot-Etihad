"""FastAPI backend. Run: uvicorn etihad_bot.api.main:app --port 8000

Sign-in is required for everything except /auth/login and /health. Identity comes from the
Authorization header (Bearer <token> from /auth/login), never from the request body.
"""
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from etihad_bot.config import get_settings
from etihad_bot.services import auth, store
from etihad_bot.services import chat as chat_service

app = FastAPI(title="Etihad Policy & Flight Assistant")


class Identity(BaseModel):
    user_id: str


def identity(authorization: str | None = Header(None)) -> Identity:
    """Reject the request unless it carries a valid, unexpired sign-in token."""
    scheme, _, token = (authorization or "").partition(" ")
    uid = store.user_for_token(token) if scheme.lower() == "bearer" and token else None
    if not uid:
        raise HTTPException(401, "Please sign in with your PNR to use the assistant.")
    return Identity(user_id=uid)


class LoginRequest(BaseModel):
    pnr: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = None


class FeedbackRequest(BaseModel):
    message_id: str
    rating: int = Field(description="1 = thumbs up, -1 = thumbs down")
    comment: str = ""


# ---- auth ----------------------------------------------------------------------------------
@app.post("/auth/login")
def login(req: LoginRequest):
    if err := auth.validate_pnr(req.pnr):  # exact match: no trimming or case folding
        raise HTTPException(422, err)
    user_id = auth.user_id_for_pnr(req.pnr)
    return {"token": store.create_session(user_id, get_settings().session_ttl_seconds),
            "user_id": user_id}


@app.post("/auth/logout")
def logout(authorization: str | None = Header(None)):
    if authorization:
        store.delete_session(authorization.partition(" ")[2])
    return {"ok": True}


# ---- chat (sync endpoints run in the thread pool; the MCP client needs its own event loop) --
@app.post("/chat")
def chat(req: ChatRequest, who: Identity = Depends(identity)):
    try:
        return chat_service.chat(who.user_id, req.message, req.conversation_id)
    except chat_service.NotOwner:
        raise HTTPException(403, "This conversation belongs to another user.")


@app.get("/conversations")
def list_conversations(who: Identity = Depends(identity)):
    return store.list_conversations(who.user_id)


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str, who: Identity = Depends(identity)):
    try:
        return {"conversation_id": conversation_id,
                "messages": chat_service.history(who.user_id, conversation_id)}
    except chat_service.NotOwner:
        raise HTTPException(403, "This conversation belongs to another user.")


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, who: Identity = Depends(identity)):
    try:
        if not chat_service.delete_conversation(who.user_id, conversation_id):
            raise HTTPException(404, "Conversation not found.")
    except chat_service.NotOwner:
        raise HTTPException(403, "This conversation belongs to another user.")
    return {"ok": True}


@app.post("/feedback")
def feedback(req: FeedbackRequest, who: Identity = Depends(identity)):
    if req.rating not in (1, -1):
        raise HTTPException(422, "rating must be 1 or -1")
    if store.message_owner(req.message_id) != who.user_id:
        raise HTTPException(404, "unknown message_id")
    store.save_feedback(message_id=req.message_id, rating=req.rating, comment=req.comment)
    return {"ok": True}


@app.get("/health")
def health():
    return {"status": "ok"}
