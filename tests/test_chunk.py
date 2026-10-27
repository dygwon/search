from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
)
from locus_ingest.chunk import chunk_document

from tests.fakes import WhitespaceTokenizer, docling_doc, table_data


def test_a_short_document_is_one_chunk_spanning_its_text():
    doc = docling_doc(text="Some body text.")

    [chunk] = chunk_document(doc, WhitespaceTokenizer())

    assert (chunk.ordinal, chunk.char_start, chunk.char_end) == (0, 0, 15)
    assert chunk.text == "Some body text."
    assert chunk.modality == "text"


def test_a_split_paragraph_gets_one_narrowed_span_per_chunk():
    para = " ".join(f"word{i}" for i in range(300))
    doc = docling_doc(heading="Long", text=para)
    # The document text joins serialized items with a blank line.
    document_text = "## Long\n\n" + para

    chunks = chunk_document(doc, WhitespaceTokenizer(max_tokens=100))

    assert len(chunks) > 1
    for chunk in chunks:
        span_text = document_text[chunk.char_start : chunk.char_end]
        assert chunk.text == f"Long\n{span_text}"
    assert [c.char_start for c in chunks] == sorted({c.char_start for c in chunks})
    assert chunks[0].char_start == len("## Long\n\n")
    assert chunks[-1].char_end == len(document_text)


def test_merged_neighbours_get_one_span_across_both_items():
    doc = docling_doc(heading="Intro", text="Short one.")
    doc.add_text(label=DocItemLabel.TEXT, text="Short two.")
    document_text = "## Intro\n\nShort one.\n\nShort two."

    [chunk] = chunk_document(doc, WhitespaceTokenizer())

    assert document_text[chunk.char_start : chunk.char_end] == "Short one.\n\nShort two."
    assert chunk.text == "Intro\nShort one.\nShort two."


def test_a_chunk_takes_page_and_bbox_from_its_first_item():
    doc = DoclingDocument(name="sample")
    bbox = BoundingBox(l=72, t=700, r=300, b=680, coord_origin=CoordOrigin.BOTTOMLEFT)
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="On page two.",
        prov=ProvenanceItem(page_no=2, bbox=bbox, charspan=(0, 12)),
    )

    [chunk] = chunk_document(doc, WhitespaceTokenizer())

    assert chunk.page == 2
    assert chunk.bbox == {"l": 72, "t": 700, "r": 300, "b": 680, "coord_origin": "BOTTOMLEFT"}


def test_a_document_without_provenance_has_no_page():
    [chunk] = chunk_document(docling_doc(), WhitespaceTokenizer())

    assert (chunk.page, chunk.bbox) == (None, None)


def test_list_items_and_tables_are_located_as_the_chunker_renders_them():
    doc = docling_doc(heading="Fruit", text="Prices below.")
    group = doc.add_list_group()
    doc.add_list_item("Apples", parent=group)
    doc.add_table(data=table_data([["fruit", "price"], ["pear", "2"]]))
    document_text = "## Fruit\n\nPrices below.\n\n- Apples\n\npear, price = 2"

    [chunk] = chunk_document(doc, WhitespaceTokenizer())

    assert document_text[chunk.char_start : chunk.char_end] == (
        "Prices below.\n\n- Apples\n\npear, price = 2"
    )


def test_token_count_covers_the_embedded_string_and_special_tokens():
    doc = docling_doc(heading="Intro", text="Short one.")
    doc.add_text(label=DocItemLabel.TEXT, text="Short two.")

    [chunk] = chunk_document(doc, WhitespaceTokenizer(), special_tokens=2)

    # "Intro\nShort one.\nShort two." is five words, plus [CLS] and [SEP].
    assert chunk.token_count == 7


def test_a_chunk_that_starts_with_a_table_has_table_modality():
    doc = DoclingDocument(name="sample")
    doc.add_table(data=table_data([["fruit", "price"], ["pear", "2"]]))

    [chunk] = chunk_document(doc, WhitespaceTokenizer())

    assert chunk.modality == "table"


def test_no_chunk_exceeds_the_budget_counting_special_tokens():
    doc = DoclingDocument(name="long")
    for section in range(4):
        doc.add_heading(f"Section {section} on quarterly revenue")
        for para in range(3):
            words = " ".join(f"s{section}p{para}w{i}" for i in range(120))
            doc.add_text(label=DocItemLabel.TEXT, text=words)

    chunks = chunk_document(doc, WhitespaceTokenizer(max_tokens=62), special_tokens=2)

    assert len(chunks) > 4
    assert max(c.token_count for c in chunks) <= 64
