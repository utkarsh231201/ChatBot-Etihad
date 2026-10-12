"""LangGraph nodes. Each returns a partial state update."""
import json
import re
import uuid
from datetime import date

from langchain_core.messages import AIMessage, HumanMessage

from etihad_bot.config import get_settings
from etihad_bot.graph import grounding
from etihad_bot.graph import guardrails as g
from etihad_bot.graph import prompts as P
from etihad_bot.graph.state import ChatState
from etihad_bot.mcp_server import client as mcp_client
from etihad_bot.rag.pipeline import Chunk, RAGPipeline
from etihad_bot.services import llm, store

HISTORY_TURNS = 6  # messages fed to the rewriter


class Nodes:
    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        self.s = get_settings()
        self._tools_desc: str | None = None

    # ---- 1. start: bind logging ctx, reset per-turn fields ------------------------------
    def start(self, state: ChatState) -> dict:
        llm.request_ctx.set({"user_id": state.get("user_id", "anonymous"),
                             "conversation_id": state.get("conversation_id", "none")})
        question = next((m.content for m in reversed(state["messages"])
                         if isinstance(m, HumanMessage)), "")
        return {"question": question, "rewritten_query": "", "intent": "", "candidates": [],
                "chunks": [], "soft_unsupported": False, "tool_result": None, "answer": "", "citations": [], "status": "",
                "message_id": uuid.uuid4().hex}

    # ---- 2. query rewriter (multi-turn memory) ------------------------------------------
    def rewrite(self, state: ChatState) -> dict:
        history = state["messages"][:-1][-HISTORY_TURNS:]
        if not history:
            return {"rewritten_query": state["question"]}
        convo = "\n".join(f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                          for m in history)
        out = llm.complete(P.REWRITE_SYSTEM,
                           f"Conversation:\n{convo}\n\nLatest user message: {state['question']}",
                           purpose="rewrite", max_tokens=200).text.strip()
        return {"rewritten_query": out or state["question"]}

    # ---- 3. intent router + clarification -----------------------------------------------
    def route(self, state: ChatState) -> dict:
        q = state["rewritten_query"]
        if g.is_self_harm_risk(state["question"]) or g.is_self_harm_risk(q):
            return {"intent": "self_harm", "answer": g.SUPPORT_MESSAGE, "status": "support"}
        if g.is_harmful_request(state["question"]) or g.is_harmful_request(q):
            return {"intent": "harmful", "answer": g.REFUSAL_HARMFUL, "status": "refused"}
        if g.is_approval_request(state["question"]) or g.is_approval_request(q):
            return {"intent": "approval_request", "answer": g.REFUSAL_APPROVAL, "status": "refused"}
        r = llm.complete_json(P.ROUTER_SYSTEM, f"Query: {q}", purpose="route", max_tokens=200)
        intent = r.get("intent") if r.get("intent") in {"flight", "policy", "smalltalk",
                                                         "unsupported", "harmful",
                                                         "self_harm", "fares"} else "policy"
        clar = (r.get("clarification") or "").strip()
        # Flight clarifications come from the flight planner (it knows the tool arguments).
        if clar and intent == "policy":
            return {"intent": intent, "answer": clar, "status": "clarification"}
        if intent == "fares":
            return {"intent": intent, "answer": g.REFUSAL_FARES, "status": "refused"}
        if intent == "self_harm":
            return {"intent": intent, "answer": g.SUPPORT_MESSAGE, "status": "support"}
        if intent == "harmful":
            return {"intent": intent, "answer": g.REFUSAL_HARMFUL, "status": "refused"}
        if intent == "unsupported":
            # Don't trust one classification call to block a topic the documents may cover:
            # search anyway and refuse only if nothing clearly relevant comes back.
            return {"intent": "policy", "soft_unsupported": True}
        if intent == "smalltalk":
            return {"intent": intent, "answer": g.GREETING, "status": "answered"}
        return {"intent": intent}

    # ---- 4a. flight path: MCP tools ------------------------------------------------------
    def flight_tool(self, state: ChatState) -> dict:
        if self._tools_desc is None:
            self._tools_desc = "\n".join(
                f"- {t['name']}: {t['description']} schema={json.dumps(t['schema'].get('properties', {}))}"
                for t in mcp_client.list_tools())
        plan = llm.complete_json(P.FLIGHT_SYSTEM.format(today=date.today().isoformat(),
                                                        tools=self._tools_desc),
                                 f"Query: {state['rewritten_query']}", purpose="flight_plan")
        if plan.get("missing") or not plan.get("tool"):
            return {"answer": plan.get("missing") or "Could you share the flight number or the "
                    "origin and destination airports?", "status": "clarification"}
        result = mcp_client.call_tool(plan["tool"], plan.get("args") or {})
        if not result.get("ok"):
            return {"tool_result": result, "answer": g.TOOL_ERROR, "status": "tool_error"}
        return {"tool_result": {"tool": plan["tool"], "args": plan.get("args"), **result}}

    def flight_answer(self, state: ChatState) -> dict:
        out = llm.complete(P.FLIGHT_ANSWER_SYSTEM,
                           f"Question: {state['rewritten_query']}\n\nTool result:\n"
                           f"{json.dumps(state['tool_result'])}", purpose="flight_answer").text
        tr = state["tool_result"]
        return {"answer": out.strip(), "status": "answered",
                "citations": [{"type": "tool", "tool": tr["tool"], "args": tr.get("args")}]}

    # ---- 4b. policy path: retrieve -> rerank -> relevance gate -> generate -> verify ----------
    @staticmethod
    def _no_answer(state: ChatState, default: str) -> str:
        """Refusal wording: out-of-scope wording if the router doubted the topic, else safety/default."""
        if state.get("soft_unsupported"):
            return g.REFUSAL_UNSUPPORTED
        return g.no_info_message(state["rewritten_query"], default)

    def retrieve(self, state: ChatState) -> dict:
        cands = self.pipeline.search(state["rewritten_query"])
        return {"candidates": [c.to_dict() for c in cands]}

    def rerank(self, state: ChatState) -> dict:
        cands = [Chunk.from_dict(d) for d in state["candidates"]]
        top = self.pipeline.rerank(state["rewritten_query"], cands)
        soft = state.get("soft_unsupported", False)
        if not self.pipeline.is_relevant(top) or (soft and top[0].score < self.s.soft_min_score):
            return {"chunks": [c.to_dict() for c in top],
                    "answer": self._no_answer(state, g.REFUSAL_NO_CONTEXT), "status": "refused"}
        return {"chunks": [c.to_dict() for c in top]}

    def generate(self, state: ChatState) -> dict:
        chunks = [Chunk.from_dict(d) for d in state["chunks"]]
        context = "\n\n".join(
            f"[{i}] ({c.document} | Section: {c.section} | Page: {c.page or 'n/a'})\n{c.text}"
            for i, c in enumerate(chunks, 1))
        out = llm.complete(P.RAG_SYSTEM, f"Context passages:\n{context}\n\n"
                           f"Question: {state['rewritten_query']}",
                           purpose="generate", max_tokens=800).text.strip()
        if "INSUFFICIENT_CONTEXT" in out:
            return {"answer": self._no_answer(state, g.REFUSAL_NO_CONTEXT), "status": "refused"}

        used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", out) if 1 <= int(n) <= len(chunks)})
        if not used:  # uncited answers are not allowed
            return {"answer": self._no_answer(state, g.REFUSAL_UNGROUNDED), "status": "refused"}

        if self.s.enable_grounding_check:
            bad = self._unsupported(out, context)
            if bad:
                out = self._repair(out, bad, context, state["rewritten_query"])
                if not out:
                    return {"answer": self._no_answer(state, g.REFUSAL_UNGROUNDED), "status": "refused"}
                used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", out)
                               if 1 <= int(n) <= len(chunks)})

        cites = [{"ref": i, **chunks[i - 1].citation(), "snippet": chunks[i - 1].text[:240]}
                 for i in used]
        return {"answer": out, "citations": cites, "status": "answered"}

    @staticmethod
    def _unsupported(answer: str, context: str) -> list[str]:
        """Claims in the answer that lack a quote verifiably present in the passages."""
        r = llm.complete_json(P.GROUNDING_SYSTEM, f"ANSWER:\n{answer}\n\nPASSAGES:\n{context}",
                              purpose="grounding_check", max_tokens=1200)
        return grounding.unsupported_claims(r.get("claims") or [], context)

    def _repair(self, answer: str, bad_claims: list[str], context: str, question: str) -> str | None:
        """Drop unsupported claims once and re-verify; None if it can't be made fully grounded."""
        named = (f"These claims in the draft are NOT supported by the passages: {json.dumps(bad_claims)}\n"
                 if bad_claims else
                 "A reviewer found part of the draft unsupported by the passages without saying which.\n")
        fixed = llm.complete(
            P.RAG_SYSTEM,
            f"Context passages:\n{context}\n\nQuestion: {question}\n\nDraft answer:\n{answer}\n\n"
            f"{named}Rewrite the answer keeping only what the passages explicitly state, dropping "
            "inferences and outside advice, with [n] citations. If nothing remains, reply "
            "INSUFFICIENT_CONTEXT.", purpose="repair", max_tokens=800).text.strip()
        if "INSUFFICIENT_CONTEXT" in fixed or not re.search(r"\[\d+\]", fixed):
            return None
        return None if self._unsupported(fixed, context) else fixed

    # ---- 6. finalize: output guardrailand persist the interaction ---------------------
    def finalize(self, state: ChatState) -> dict:
        answer, status = state["answer"], state["status"]
        if status == "answered" and g.violates_output_policy(answer):
            answer, status = g.REFUSAL_APPROVAL, "refused"
        store.save_interaction(
            message_id=state["message_id"], user_id=state.get("user_id", "anonymous"),
            conversation_id=state.get("conversation_id", "none"), question=state["question"],
            rewritten_query=state["rewritten_query"], intent=state.get("intent", ""),
            answer=answer, status=status,
            chunks=state.get("chunks", []), citations=state.get("citations", []))
        return {"answer": answer, "status": status, "messages": [AIMessage(content=answer)]}


