from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class ChatState(TypedDict, total=False):
    # persisted across turns by the checkpointer
    messages: Annotated[list[AnyMessage], add_messages]

    # identifiers
    user_id: str
    conversation_id: str

    # per-turn fields (reset by the `start` node)
    question: str
    rewritten_query: str
    intent: str  # policy | flight | smalltalk | unsupported | approval_request
    soft_unsupported: bool  # router doubted the topic: search anyway, but demand a clearly relevant hit
    candidates: list[dict]
    chunks: list[dict]  # final reranked chunks
    tool_result: Optional[dict]
    answer: str
    citations: list[dict]
    status: str  # answered | clarification | refused | tool_error
    message_id: str

