from docling_core.types.doc import DocItemLabel, DoclingDocument
from locus_core.models import ResolvedModel

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
