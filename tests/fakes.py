import hashlib
import math
import random
import zlib
from collections import Counter

from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.types.doc import DocItemLabel, DoclingDocument, TableCell, TableData
from locus_core.models import ResolvedModel
from locus_core.protocols import SparseVector

BGE_SMALL = ResolvedModel(
    repo_id="BAAI/bge-small-en-v1.5",
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    max_seq_length=512,
    dim=384,
)


class FakeResolver:
    """Stands in for the Hugging Face Hub."""

    def __init__(self, *models):
        self.models = {model.repo_id: model for model in models}

    def resolve(self, repo_id):
        return self.models[repo_id]


class FakeParser:
    """Stands in for Docling: returns a prepared document, or raises."""

    version = "fake/1"

    def __init__(self, doc=None, error=None):
        self.doc = doc if doc is not None else docling_doc()
        self.error = error
        self.calls = 0

    def parse(self, path):
        self.calls += 1
        if self.error:
            raise self.error
        return self.doc


def docling_doc(heading=None, text="Some body text."):
    doc = DoclingDocument(name="sample")
    if heading:
        doc.add_heading(heading)
    doc.add_text(label=DocItemLabel.TEXT, text=text)
    return doc


def write_pdf(path, text="Hello Locus", title=None):
    """Write a one-page PDF, with a metadata title when given."""
    stream = f"BT /F1 24 Tf 72 700 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
        b" /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    if title is not None:
        objects.append(f"<< /Title ({title}) >>".encode())
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    info = b" /Info %d 0 R" % len(objects) if title is not None else b""
    out += b"trailer\n<< /Size %d /Root 1 0 R%s >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        info,
        xref,
    )
    path.write_bytes(bytes(out))
    return path


class WhitespaceTokenizer(BaseTokenizer):
    """Stands in for a Hugging Face tokenizer: one token per whitespace-separated word."""

    max_tokens: int = 512

    def count_tokens(self, text):
        return len(text.split())

    def get_max_tokens(self):
        return self.max_tokens

    def get_tokenizer(self):
        return self.count_tokens


def table_data(rows):
    """A table whose first row is the column header."""
    cells = [
        TableCell(
            text=text,
            start_row_offset_idx=r,
            end_row_offset_idx=r + 1,
            start_col_offset_idx=c,
            end_col_offset_idx=c + 1,
            column_header=r == 0,
        )
        for r, row in enumerate(rows)
        for c, text in enumerate(row)
    ]
    return TableData(num_rows=len(rows), num_cols=len(rows[0]), table_cells=cells)


class FakeDenseEmbedder:
    """Stands in for sentence-transformers: unit vectors seeded by the text's hash."""

    def __init__(self, dim=384, fail_with=None):
        self.dim = dim
        self.fail_with = fail_with
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        if self.fail_with:
            raise self.fail_with
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        rng = random.Random(hashlib.sha256(text.encode()).digest())
        vector = [rng.gauss(0, 1) for _ in range(self.dim)]
        norm = math.sqrt(sum(x * x for x in vector))
        return [x / norm for x in vector]


class FakeSparseEmbedder:
    """Stands in for FastEmbed BM25: one term frequency per hashed word."""

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        counts = Counter(zlib.crc32(word.lower().encode()) % 100_000 for word in text.split())
        indices = sorted(counts)
        return SparseVector(indices=indices, values=[float(counts[i]) for i in indices])
