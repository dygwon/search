# Chunk spans are item-level

A chunk's span in the **document text** is worked out from the items it covers, not by finding its rendered text. Matching rendered text meant copying how Docling's chunker renders each case, and real papers broke it in ways that can't be patched: a caption has no text of its own in the document text (it renders as the start of its picture's or table's text), a chunk holding a caption and its table counts the caption twice, a caption renders as empty text when its table falls in the next chunk, and a table over the token budget is split into markdown rows (`| - |`) that appear nowhere in the document text.

The rule:

- An item without a span of its own (a caption) resolves to the nearest ancestor that has one (its picture or table). A chunk's resolved items are de-duplicated in order.
- A chunk whose only resolved item is the text item it lists, such as a piece of a split paragraph, gets a narrowed span: where its text is found in that item, after the previous chunk's end.
- Any other chunk spans from the start of its first resolved item to the end of its last.
- If a chunk has an item that resolves to nothing, or its text isn't found in its one text item, the chunk stage fails rather than guess a span.

Several chunks can now share a span (the pieces of a split table). So `chunk_id` is the UUID5 of doc hash, config hash, span start, span end and **piece**: how many earlier chunks of the document have the same span. Piece is 0 for nearly every chunk. Chunking is deterministic, so piece is too. The config hash includes an offsets-scheme marker, so IDs made under this rule never equal IDs made under ADR 0001's.

## Consequences

- Spans are coarser for chunks that don't fall inside a single text item, and they can overlap. A chunk that starts partway through a paragraph and runs into the next item spans the whole paragraph. Nothing relies on spans being disjoint.
- A piece of a split table can't be pointed at more precisely than its whole table.
- Chunking no longer depends on how Docling renders captions or split tables, so a Docling upgrade is less likely to break ingest.
- Indexes built before this change keep their old chunk IDs until re-chunked. There's no `index rm` or re-chunk in place yet, so to bring one in line, ingest into a fresh index. Parses are cached by doc hash, so Docling doesn't run again.
