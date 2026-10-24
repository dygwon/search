# Chunk offsets point into the document text

A chunk's `char_start`/`char_end`, and so its `chunk_id`, are offsets into the **document text**: every item of the parsed Docling document, in reading order, rendered by the chunker's own serializer and joined with a blank line (`"\n\n"`). We render with the chunker's serializer, not `item.text`, because the chunker emits list markers (`- Apples`) and table triplets (`pear, price = 2`) that `item.text` lacks or doesn't have at all. Every chunk can then be found in the document text, and the chunk stage fails rather than guess a span, since a guessed span could give two chunks one ID. Plain text items render as their own text, so their offsets don't depend on chunk size or index. Lists, tables and headings carry the serializer's markup, so their offsets can shift with a Docling upgrade.

## Consequences

- The same document chunked with the same config gets the same `chunk_id` in every index, so `chunks` is keyed by `(index_id, chunk_id)`, not by `chunk_id` alone as the design doc's table says. Qdrant points are per collection, so they're unaffected.
- A chunk's span starts at its text, not at the heading prefix the chunker adds; the prefix belongs to the embedded string, not to the document text.
