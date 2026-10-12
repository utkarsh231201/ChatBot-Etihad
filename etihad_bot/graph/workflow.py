"""LangGraph workflow.

start -> rewrite -> route -+-> finalize                              (clarify / refuse / smalltalk)
                           +-> flight_tool -> flight_answer -> finalize
                           `-> retrieve -> rerank -> generate -> finalize
"""
import sqlite3
from functools import lru_cache

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from etihad_bot.config import get_settings
from etihad_bot.graph.nodes import Nodes
from etihad_bot.graph.state import ChatState
from etihad_bot.rag.pipeline import RAGPipeline, get_pipeline


def _after_route(state: ChatState) -> str:
    if state.get("status"):  # clarification / refusal / smalltalk already decided
        return "finalize"
    return "flight_tool" if state["intent"] == "flight" else "retrieve"


def _after_flight_tool(state: ChatState) -> str:
    return "finalize" if state.get("status") else "flight_answer"


def _after_rerank(state: ChatState) -> str:
    return "finalize" if state.get("status") == "refused" else "generate"


def build_graph(pipeline: RAGPipeline | None = None, checkpointer=None):
    n = Nodes(pipeline or get_pipeline())
    g = StateGraph(ChatState)
    for name in ("start", "rewrite", "route", "flight_tool", "flight_answer",
                 "retrieve", "rerank", "generate", "finalize"):
        g.add_node(name, getattr(n, name))

    g.set_entry_point("start")
    g.add_edge("start", "rewrite")
    g.add_edge("rewrite", "route")
    g.add_conditional_edges("route", _after_route, ["finalize", "flight_tool", "retrieve"])
    g.add_conditional_edges("flight_tool", _after_flight_tool, ["finalize", "flight_answer"])
    g.add_edge("flight_answer", "finalize")
    g.add_edge("retrieve", "rerank")
    g.add_conditional_edges("rerank", _after_rerank, ["finalize", "generate"])
    g.add_edge("generate", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


@lru_cache(maxsize=1)
def get_checkpointer() -> SqliteSaver:
    conn = sqlite3.connect(get_settings().checkpoint_db, check_same_thread=False)
    return SqliteSaver(conn)


@lru_cache(maxsize=1)
def get_app():
    """Production graph: persistent SQLite checkpointer, thread_id = conversation_id."""
    return build_graph(checkpointer=get_checkpointer())
