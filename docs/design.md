# Locus — Design Doc

Oct 23, 2026 · @Daniel Gwon

## Goals and constraints

A local, single-user, search-first RAG app for sensitive documents; generation is optional and runs locally by default.

- Documents never leave the machine; indexes read files in place and never copy them.
- Users pick the files to index; nothing watches the filesystem.
- Generation uses a local open model via base URL and API key; a proprietary API is an explicit opt-in.
- PDF first; Word, PowerPoint, Excel, tables, figures, and images later, so the schema is multimodal from day one.
- Hybrid retrieval (BM25, dense, SPLADE) with ColBERT reranking; the pipeline lives in app code, not the database.
- Runs on Apple silicon (Metal/MPS) and NVIDIA (CUDA).
- Personal use, so non-commercial model licenses are acceptable.
- Evaluation comes later; opt-in run logging writes TREC-format runs for trec\_eval.

## Architecture overview

Two modules, ingest and search, share one set of on-demand model servers and two local stores; only opt-in generation leaves the machine.

> **Diagram missing.** The architecture diagram (9 components, one optional remote call) didn't survive the Markdown export from the original doc. Re-add it here.

Clients talk only to the API, which runs searches and queues ingest jobs for the worker. Qdrant runs in Docker because it needs no GPU; everything that touches the GPU runs natively.

## Data model and identity

SQLite in WAL mode is the system of record; Qdrant is a derived index that can be rebuilt from it.

| Table | Key | Holds |
| --- | --- | --- |
| documents | doc\_hash | source path, title, title source, MIME type, size, added\_at |
| parse\_cache | doc\_hash + parser\_version | path to the cached Docling JSON |
| indexes | index\_id | name, model bundle with pinned revisions, chunk target and ceiling, chunker version, Qdrant collection and alias, status |
| index\_members | index\_id + source\_path | current doc\_hash, status, updated\_at |
| chunks | index\_id + chunk\_id ([ADR 0001](adr/0001-chunk-offsets.md)) | doc\_hash, ordinal, char\_start, char\_end, page, bbox, modality, text, token count per model |
| jobs | job\_id | index\_id, doc\_hash, stage, status, attempts, error, updated\_at |
| model\_servers | server\_id | model ID, revision, kind, port, PID, status, last\_used |
| query\_log | query\_id | index\_id, query text, per-stage ranked lists (only when run logging is on), timestamp |

**Identity**

- doc\_hash = SHA-256 of the file bytes.
- chunk\_id = UUID5 of (doc\_hash, chunker config hash, char\_start, char\_end); it doubles as the Qdrant point ID.
- Character offsets let relevance judgments survive re-chunking and apply across indexes.

**Lifecycle**

- Re-adding a path with a new hash replaces the old version. The new version is ingested and upserted first, then the old chunks and points are deleted, so search never sees the document missing.
- Deleting a document removes its index\_members row, its chunks, and its Qdrant points (filtered on doc\_hash) in one job.
- Judgments that reference a replaced doc\_hash are flagged as stale.
- The ingest worker is the only heavy writer. The API writes only query logs and status updates, with a busy timeout.

## Index definition and Qdrant schema

An index is an immutable bundle of models and chunk settings chosen at creation; changing any model means creating a new index.

**Model bundle**

- Dense, SPLADE, and ColBERT models, plus an optional cross-encoder, each pinned to a Hugging Face commit hash.
- BM25 needs no model.
- Search always loads the exact revisions used at ingest.

**Chunk budget**

- Ceiling = the smallest max sequence length across the bundle, counted in each model's own tokenizer. ColBERT uses its configured document length.
- Target = a user-set size, default 256–512 tokens, capped by the ceiling.
- The budget counts the final embedded string, including any heading prefix the chunker adds.
- Any truncation is logged as an error.

**Qdrant collection** (one per index; alias = index name)

| Field | Kind | Config |
| --- | --- | --- |
| dense | dense vector | size from the model, cosine, HNSW |
| splade | sparse vector | no modifier; learned weights already encode term importance |
| bm25 | sparse vector | IDF modifier; term frequencies from FastEmbed |
| colbert | multivector | MaxSim, HNSW disabled (m=0), rerank only |

- Payload: chunk\_id, doc\_hash, title, page, bbox, modality, char\_start, char\_end, text, filter fields.
- Payload indexes: doc\_hash and modality (keyword), title (full text), added\_at (datetime).
- Adding a new named vector later may need a new collection plus an alias swap. Stored ColBERT vectors mean a rebuild never re-encodes documents.

## Ingestion pipeline

Ingestion runs as resumable per-document jobs: parsing happens once per document, chunking and embedding once per index.

1. **Register.** Hash the file, upsert documents and index\_members rows, and resolve the title: PDF metadata, then the first Docling heading, then the filename. Titles are user-editable.
2. **Parse.** Docling writes JSON with page and bbox provenance, cached by doc\_hash and parser version. A cache hit skips this stage.
3. **Chunk.** HybridChunker with the strictest tokenizer in the bundle; enforce the ceiling and write chunks rows.
4. **Embed.** Dense and SPLADE via TEI, BM25 term frequencies via FastEmbed, ColBERT via PyLate. Batches are sorted by token length.
5. **Upsert.** Write Qdrant points with deterministic IDs, then retire the previous version when replacing.

The jobs table records state per document per stage, so a crash or cancel resumes at the last finished stage. Progress streams to the UI over SSE.

**Speed tactics**

- Profile seconds per page by stage before optimizing; Docling often costs more than embedding.
- Parse in a CPU process pool while the GPU embeds the previous batch.
- Skip chunks whose text hash was already embedded with the same model.
- Run bulk ColBERT encoding in-process; per-token vectors are large as HTTP JSON.

## Search pipeline

Each search targets one index and runs five stages in app code; logging each stage's ranked list is opt-in.

1. **Encode.** Encode the query with the index's bundle: dense, SPLADE, BM25 term frequencies, and ColBERT query vectors.
2. **Retrieve.** One `query_batch_points` call returns separate dense, SPLADE, and BM25 lists, 100 candidates each by default, with filters applied.
3. **Fuse.** A pluggable `Fuser`: RRF (k=60, the default), weighted RRF, DBSF, or a convex combination. A chunk missing from a list gets an explicit default score.
4. **Rerank.** ColBERT MaxSim over the top 50, run in Qdrant on the `colbert` field filtered to the candidate IDs (`HasIdCondition`). An optional cross-encoder via TEI `/rerank` then reorders the top 20.
5. **Present.** Group hits by doc\_hash so one document can't fill the page. Each hit shows title, page, snippet, and per-stage scores.

Optional generation runs over the top chunks. Candidate depths are starting points to tune; each list's depth bounds what later stages can recover.

## Generation

Generation is optional and local by default; a remote provider needs an explicit opt-in each session.

- A `Generator` protocol with two adapters: OpenAI-compatible (base URL and API key, covering vLLM, Ollama, llama.cpp, and LM Studio) and native Anthropic.
- Configuration through pydantic-settings; API keys are never logged.
- Remote opt-in shows a persistent indicator in the UI and TUI, and logs which chunk IDs were sent.
- Answers cite chunks by document title and page; the exact format is deferred.
- Responses stream over SSE.

## Model serving

Model servers start on demand when an index is ingested or searched, run natively for GPU access, and stop when idle.

| Backend | Covers | Role |
| --- | --- | --- |
| TEI | dense, SPLADE (`/embed_sparse`), cross-encoder (`/rerank`) | primary; Metal build on Mac, CUDA on NVIDIA |
| PyLate | ColBERT document and query vectors | in-process for ingest, small service for search |
| FastEmbed | BM25 term frequencies | in-process |
| vLLM | dense, ColBERT, `/rerank`, `/score` | NVIDIA alternative |

All backends sit behind `Embedder` (dense, sparse, multivector) and `Reranker` protocols, so swapping one never touches pipeline code.

**Lifecycle manager**

- TEI serves one model per process, so one bundle needs several servers.
- On request, reuse a running server for the same model and revision from model\_servers; otherwise launch one on a free localhost port and health-check it.
- Ingest and search share servers for the same model.
- A configurable idle timeout and memory budget evict the least recently used servers.
- On startup, reconcile model\_servers rows with live PIDs.

## Interfaces, repo layout and spin-up

One uv workspace holds a shared core and two runnable modules, all behind a single `locus` CLI.

```
locus/
  packages/core     settings, SQLite schema, protocols, adapters, model-server manager
  packages/ingest   register, parse, chunk, embed, upsert jobs
  packages/search   encode, retrieve, fuse, rerank, generate
  apps/api          FastAPI + HTMX/Jinja web UI, SSE
  apps/tui          Textual client of the API
```

- **CLI:** `locus index create`, `locus ingest <files> --index <name>`, `locus serve`, and `locus up` to run everything end to end.
- **Protocols in core:** `Embedder`, `Reranker`, `Fuser`, `VectorStore`, `Generator`, `RunLogger`.
- **Libraries:** async `qdrant-client` over gRPC, `httpx`, Docling, PyLate, FastEmbed, pydantic. No LangChain or LlamaIndex.
- **Spin-up:** Qdrant runs in Docker since it needs no GPU. The API, worker, and model servers run natively. `locus up` starts Qdrant, the API, and the worker; the app launches model servers itself. process-compose or a Procfile is the default runner.

## Security and privacy

Every service binds to 127.0.0.1; nothing listens on the network.

- Map Qdrant ports as `127.0.0.1:6333:6333` and `127.0.0.1:6334:6334`. Without the address prefix, Docker exposes them on every interface.
- TEI, PyLate, and the API bind to localhost only; Qdrant gets an optional API key.
- Query logs contain sensitive text, so they stay local with a configurable retention.
- The remote LLM is off by default, opted into per session, and logged.
- Model downloads are the only expected outbound traffic, and revisions are pinned.

## Evaluation hooks and planned experiments

Evaluation is deferred, but opt-in run logging and stable IDs make it possible later without rework. Logging is off by default because run files grow quickly; users turn it on per session or with a CLI flag when they plan to evaluate.

- Every stage writes TREC run lines: query\_id, Q0, chunk\_id, rank, score, stage tag, while logging is on.
- Relevance judgments are made at document plus character-span level, then mapped to chunks per index.
- trec\_eval (or ir\_measures) scores the run files.

| Question | Metric |
| --- | --- |
| Where does ingest time go? | seconds per page by stage |
| Does SPLADE add recall beyond dense plus BM25? | recall@100 |
| RRF or a tuned convex combination? | nDCG@10 |
| ColBERT or a cross-encoder for reranking the top 50? | nDCG@10, p95 latency |
| Which chunk target size works best? | recall@100, nDCG@10 |
| Hybrid first stage or ColBERT first stage? | recall@100 |

## Deferred decisions

None of these block the first build; each can be settled without a schema change.

- [ ] Default models for each bundle slot
- [ ] Latency budget per search, which sets candidate depths and the reranker choice
- [ ] Citation format for generated answers
- [ ] Remote-LLM opt-in UX details
- [ ] Backups that snapshot SQLite and Qdrant together
- [ ] Query log retention period
- [ ] Cross-index search, which would need rank-level fusion
- [ ] ColBERT first-stage retrieval: a MUVERA vector in Qdrant, or a PLAID/WARP index
- [ ] Multimodal retrieval for figures and pages: a dense multimodal embedding model (one vector per item), or late-interaction multivectors (ColPali family, ColModernVBERT)
