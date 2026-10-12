# Etihad Policy & Flight Assistant

A conversational assistant that answers **policy questions** from an Etihad policy PDF (RAG with citations) and
**flight questions** (status / schedule / routes) through **MCP tools** backed by the AviationStack API. Built on
LangGraph with persistent memory, PNR sign-in, guardrails, logging, feedback and an evaluation + ablation framework.

## Architecture

```
 Streamlit UI  --HTTP-->  FastAPI  (/auth  /chat  /conversations  /feedback)
 (sign-in gate, 👍/👎, resume, delete)        |
                                         v
                LangGraph (SqliteSaver checkpointer, thread_id = conversation_id)

 start --> rewrite --> route ----+--> flight_tool --MCP--> flight_answer --+
 (reset)   (memory)    (intent,  |    (LLM picks tool + args)             |
                       clarify,  +--> retrieve (Chroma, top-20)           |
                       refund    |    --> rerank (bge-reranker, top-5)    +--> finalize
                       guard)    |    --> relevance gate --no--> refuse   |    (output guardrail,
                                 |    --> generate (cited) --> grounding  |     SQLite log,
                                 |        check (repair once or refuse)   |     AIMessage)
                                 +--> clarify / refuse / smalltalk -------+
```

**Retrieval flow:** User Query -> Query Rewriter -> ChromaDB retrieval -> Re-ranker -> LLM -> Answer + Citations.

**Indexing flow:** PDF loader -> text splitter (1000 / 200) -> BGE embeddings -> ChromaDB.

## Folder structure

```
etihad_bot/
  config.py                 pydantic settings (.env)
  rag/
    loaders.py              PDF + website loaders; headings detected by font size; section + page metadata
    splitting.py            RecursiveCharacterTextSplitter (1000/200) + stable chunk ids
    embeddings.py           BAAI/bge-large-en-v1.5 (normalised, query instruction)
    pipeline.py             ChromaDB store, search(), rerank() (bge-reranker-base), relevance gate
    ingest.py               CLI ingestion (policy PDF only by default)
  mcp_server/
    flight_api.py           AviationStack client (+ mock mode)
    server.py               FastMCP server: get_flight_status / get_flight_schedule / get_flight_routes
    client.py               MCP stdio client used by the graph
  graph/
    state.py  nodes.py  workflow.py   LangGraph state, nodes, wiring, checkpointer
    prompts.py  guardrails.py         prompts and deterministic guardrails
  services/
    llm.py                  OpenAI wrapper -> logs tokens/latency/cost/status per call
    store.py                SQLite: llm_calls, interactions, feedback, sessions
    auth.py                 PNR validation + pseudonymous user ids
    chat.py                 chat() / history() / delete_conversation()
  api/main.py               FastAPI
ui/app.py                   Streamlit chat: sign-in, 👍/👎, conversation resume + delete
evaluation/                 metrics.py, run_eval.py, ablation.py
data/pdfs/                  the policy PDF       data/eval/eval_set.json   gold set
tests/                      unit tests
```

## Setup

Use **Python 3.13** (what the dependencies were installed and tested on). `mcp` is pinned `<2` because 2.x renamed
`FastMCP`. Run commands from the project root.

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env            # add OPENAI_API_KEY, AVIATIONSTACK_API_KEY and a random AUTH_PEPPER
python -m etihad_bot.rag.ingest                          # indexes ONLY data/pdfs/etihad_policy1.pdf
uvicorn etihad_bot.api.main:app --port 8000
streamlit run ui/app.py
```

No flight API key? Set `FLIGHT_API_MOCK=true` for canned data. The first run downloads the BGE embedding (~1.3 GB)
and reranker models. Web-page loading exists (`--url`) but is not used by default.

### API

**Sign-in is required.** Every endpoint except `/auth/login` and `/health` returns `401` without a valid `Authorization: Bearer <token>` header, so the assistant answers nobody who has not signed in. Identity comes from that header, never the body.

| Endpoint | Purpose |
|---|---|
| `POST /auth/login` `{pnr}` | sign in; returns a 24 h token. PNR format is 2 letters + 2 digits + 2 letters (e.g. `AB23FG`), **case-sensitive** |
| `POST /auth/logout` | end the session |
| `POST /chat` `{message, conversation_id?}` | answer, citations, `message_id`, status |
| `GET /conversations` | list your conversations |
| `GET /conversations/{id}` | reload a checkpointed conversation (owner only) |
| `DELETE /conversations/{id}` | delete messages, feedback and the LangGraph checkpoint (owner only) |
| `POST /feedback` `{message_id, rating: 1\|-1, comment?}` | stores feedback + the retrieved chunks |

**Sign-in caveat:** there is no booking system behind it, so a PNR is *not verified* against Etihad. It is an identity
key: anyone who knows a PNR can sign in as it and read or delete that history. For real use add a second factor (e.g.
surname) or validate against the reservation system. The raw PNR is never stored; user ids are a salted hash
(`AUTH_PEPPER`). Deleting keeps the `llm_calls` accounting rows (tokens/cost/latency, no message text).

## Feature map

| # | Requirement | Where |
|---|---|---|
| 1 | RAG, metadata, 1000/200 splitter, BGE, Chroma, citations (document, section, page) | `rag/*`, `Nodes.generate` |
| 2 | MCP server + client, 3 flight tools, third-party API | `mcp_server/*` |
| 3 | Multi-turn memory, query rewriting | `ChatState.messages`, `Nodes.rewrite` |
| 4 | SQLite checkpointer, resume | `workflow.get_checkpointer`, `chat.history` |
| 5 | Redis cache | **Removed by choice** (see below) |
| 7 | Intent router | `Nodes.route` |
| 8 | Guardrails | `graph/guardrails.py`, `prompts.py`, relevance gate, grounding check |
| 9 | Logging (user, conversation, model, tokens, latency, cost, status) | `services/llm.py` -> `llm_calls` table |
| 10 | Feedback with retrieved chunks | `/feedback` -> `feedback` table |
| 11 | Evaluation + ablation | `evaluation/` |

**No response cache.** Redis was deliberately removed, so every question runs the full pipeline. A cache can be added
later between `rewrite` and `route`, keyed on the rewritten query; cache only cited policy answers, never flight data.

## Guardrails

| Risk | Defence |
|---|---|
| Hallucination | answer only from numbered passages; uncited answers rejected; `INSUFFICIENT_CONTEXT`; reranker-score relevance gate; verifiable grounding check: the checker must quote each claim word for word and code confirms the quote exists in the passages (one repair pass, else refuse) |
| Refund / waiver approval | input regex short-circuits to a refusal; system prompt forbids approvals; output regex blocks approval phrasing |
| Dangerous goods (lighters, perfumes, batteries, ...) with no rule in the documents | never answered from general knowledge; the bot says the documents don't cover it, that such items can be restricted, and to confirm with Etihad before packing (`SAFETY_NO_INFO`) |
| Harmful requests (evading security screening, smuggling, harming passengers or crew) | deterministic regex plus a `harmful` router class -> firm refusal. Narrow on purpose, so "is a lighter dangerous?" is still treated as a normal question |
| Possible self-harm ("I want to jump from the flight", "I want to die") | checked before the harmful-request rule; a regex plus a `self_harm` router class. The reply is a gentle check-in with crisis resources and an offer to help if they meant something else, never a security refusal (`SUPPORT_MESSAGE`, status `support`) |
| Fare, price, availability or booking requests ("cheapest flight to London") | the flight feed has no price data, so the router's `fares` class returns a clear "can't search fares" reply that points to etihad.com and offers schedules instead. Published fees in the PDF (pet fee, UM fee, Priority Access) are still answered as policy |
| Unsupported questions | a question the router calls `unsupported` is still searched against the PDF and refused only if nothing clearly relevant comes back, so product names like "Etihad Priority Access" are never blocked by one classification call |
| Missing details | router (policy) and flight planner (flights) ask one clarifying question instead of guessing |

## Logging & feedback schema (SQLite `data/sqlite/app.sqlite`)

- `llm_calls`: user_id, conversation_id, purpose, model, input_tokens, output_tokens, latency_ms, cost_usd, status.
  Cost uses `LLM_PRICE_IN_PER_M` / `LLM_PRICE_OUT_PER_M`. **Check those against current pricing.**
- `interactions`: one row per answer incl. retrieved chunks and citations.
- `feedback`: rating, comment, question, answer and the retrieved chunks joined from `interactions`.
- `sessions`: sign-in tokens.

Checkpoints live in `data/sqlite/checkpoints.sqlite`.

## Evaluation

```bash
py -3.13 -m evaluation.run_eval --skip-flight      # full-graph metrics
py -3.13 -m evaluation.ablation                    # retrieval-only ablation, no LLM cost (slow: re-embeds per chunk size)
py -3.13 -m evaluation.ablation --e2e              # + full metrics per configuration (LLM cost)
```

| Metric | Definition |
|---|---|
| Retrieval recall | share of gold `evidence_keywords` found in the final top-5 chunks (chunk-size independent); MRR also reported |
| Answer correctness | LLM judge vs. gold answer (0-1); refusing an answerable question scores 0 |
| Faithfulness | LLM judge: share of claims supported by the retrieved context |
| Citation accuracy | LLM judge: share of `[n]` citations whose passage supports the cited sentence |
| Refusal accuracy | refused <=> expected to refuse, over answer/refuse cases |
| Clarification rate / accuracy | share of responses that ask a question / whether that matched the gold label |

Latest end-to-end run (13 cases): recall 0.86, correctness 0.86, faithfulness 0.93, citation accuracy 0.93, refusal
accuracy 1.0, clarification accuracy 1.0.

Ablation (recall@5 / MRR; chunk size, overlap = 20 %):

| Chunk | Reranker off | Reranker on |
|---|---|---|
| 500 | 0.857 / 0.786 | 0.857 / **0.857** |
| 1000 | 0.857 / 0.571 | 0.857 / 0.690 |
| 1500 | **1.000** / 0.624 | 0.857 / 0.762 |
| 2000 | 0.857 / 0.595 | 0.857 / 0.762 |

The reranker improved MRR at every chunk size. With only 7 policy questions, one missed question is the whole gap between
0.857 and 1.0, so treat these as directional.

**Caveats:** the gold set is small (14 cases) and written from the first pages of the sample PDF; extend
`data/eval/eval_set.json` before drawing conclusions. The judge uses the same model family as the generator, which biases
scores upward. "Can I take my dog in First class?" is a known retrieval miss.

## Known limitations

- Flight data is live but limited by the AviationStack free plan: no date filtering, only about yesterday and today, and
  no `routes` endpoint (routes are inferred from currently tracked Etihad flights and may be incomplete). An empty result
  means "not in the live feed", not "no such flight".
- The MCP server is spawned per tool call over stdio (~1 s overhead). Use a long-lived session for production.
- Web pages have no page numbers, so their citations carry `page = 0` (shown as n/a).
- PDF headings are detected by font size; the last pages of the sample PDF (web-printed extra-baggage text) have none and
  are filed under the final section ("32. Medical travel insurance").
