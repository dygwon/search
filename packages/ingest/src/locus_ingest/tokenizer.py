"""The chunker's tokenizer: the dense model's own, at its pinned revision."""

from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from locus_core.models import ResolvedModel


def hf_tokenizer(model: ResolvedModel, budget: int) -> tuple[HuggingFaceTokenizer, int]:
    """A tokenizer for chunks of at most budget tokens, and the special tokens it leaves out.

    Docling counts tokens without [CLS]/[SEP], but the model's limit includes them,
    so they come off the budget the chunker sees.
    """
    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained(model.repo_id, revision=model.revision)
    special_tokens = hf.num_special_tokens_to_add()
    return HuggingFaceTokenizer(tokenizer=hf, max_tokens=budget - special_tokens), special_tokens
