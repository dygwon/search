import pytest
from locus_ingest.parse import DoclingParser

from tests.fakes import write_pdf

# Docling downloads its layout models from the Hugging Face Hub on first use.
pytestmark = pytest.mark.network


def test_docling_parses_text_with_page_provenance(tmp_path):
    pdf = write_pdf(tmp_path / "sample.pdf", text="Hello Locus")

    doc = DoclingParser().parse(pdf)

    [item] = [item for item in doc.texts if "Hello Locus" in item.text]
    [prov] = item.prov
    assert prov.page_no == 1
    assert prov.bbox.width > 0
