# Locus

Local, search-first retrieval over a user's own sensitive documents.

## Language

**Document**:
A file's exact bytes, identified by their SHA-256 (the doc hash). The same bytes at two paths are one document.
_Avoid_: File (a file is where a document lives), source

**Index**:
An immutable bundle of models and chunk settings, chosen once, that documents are added to and searched through.
_Avoid_: Collection (that's Qdrant's storage for an index), corpus

**Document text**:
The parsed document as one string: each item, in reading order, as the chunker renders it, separated by blank lines. Chunk offsets point into it.
_Avoid_: Full text, raw text, markdown

**Chunk**:
A contiguous span of the document text, sized to fit the index's models, that is embedded and retrieved as one unit.
_Avoid_: Passage, segment, split
